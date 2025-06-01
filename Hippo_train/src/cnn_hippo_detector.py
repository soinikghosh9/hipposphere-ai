# src/cnn_hippo_detector.py
import cv2
import numpy as np
import os
import json
from datetime import datetime # For unique timestamps with microseconds
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam # Or AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging

from src import config # Import HippoSphereAI's config

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Module-level variables ---
module_loaded_cnn_model = None 

open_kernel_bg_cnn, close_kernel_bg_cnn, open_kernel_fd_cnn, close_kernel_fd_cnn = None, None, None, None

# State for the bounding box annotation mode (run_cnn_bbox_annotation_mode functions)
_cnn_annot_in_annotation_mode = False
_cnn_annot_annotating_id_pending = None # Stores 0 (BG), 1 (H1), or 2 (H2) etc. if selected for drawing
_cnn_annot_current_roi_points = [] # Stores [(x1,y1), (x2,y2)] for the ROI being drawn
_cnn_annot_temp_frame_for_annotation = None # A static copy of the frame when an ID is selected for drawing ROI
_cnn_annot_current_video_path_global = ""
_cnn_annot_current_frame_num_global = 0
_cnn_annot_annotations_data = [] # List to store all annotation dicts

_cnn_annot_active_trackers = {} # {hippo_id: tracker_object} for hippo profiles
_cnn_annot_current_bboxes = {}  # {hippo_id: (x,y,w,h)} for hippo profiles (persistent display from trackers)


def _initialize_kernels_cnn():
    """Initializes morphological kernels from config values."""
    global open_kernel_bg_cnn, close_kernel_bg_cnn, open_kernel_fd_cnn, close_kernel_fd_cnn
    if open_kernel_bg_cnn is None: # Check one to see if all need init
        open_kernel_bg_cnn = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_OPEN_BG_KERNEL_SIZE_CNN)
        close_kernel_bg_cnn = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_CLOSE_BG_KERNEL_SIZE_CNN)
        open_kernel_fd_cnn = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_OPEN_FD_KERNEL_SIZE_CNN)
        close_kernel_fd_cnn = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_CLOSE_FD_KERNEL_SIZE_CNN)
        logger.debug("Morphological kernels for CNN detector initialized.")

def load_trained_cnn_model(): 
    """Loads the trained Keras CNN model from path in config, updates module-level var."""
    global module_loaded_cnn_model
    if not hasattr(config, 'CNN_MODEL_SAVE_PATH') or not config.CNN_MODEL_SAVE_PATH:
        logger.error("config.CNN_MODEL_SAVE_PATH is not defined.")
        return False
    model_path = config.CNN_MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        logger.warning(f"Trained CNN model not found at {model_path}. Please train it first (Option 1C).")
        module_loaded_cnn_model = None
        return False
    try:
        module_loaded_cnn_model = load_model(model_path)
        logger.info(f"Successfully loaded module-level trained CNN model from: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading CNN model from {model_path}: {e}", exc_info=True)
        module_loaded_cnn_model = None
        return False

def create_cv_tracker_cnn(tracker_type_str_local=None):
    """Creates an OpenCV tracker instance based on the type specified in config."""
    if tracker_type_str_local is None:
        tracker_type_str_local = getattr(config, 'TRACKER_TYPE_CV_CNN', 'CSRT')

    try:
        if tracker_type_str_local.upper() == 'CSRT': return cv2.TrackerCSRT_create()
        elif tracker_type_str_local.upper() == 'KCF': return cv2.TrackerKCF_create()
        elif tracker_type_str_local.upper() == 'MOSSE': 
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                return cv2.legacy.TrackerMOSSE_create()
            else:
                logger.warning("cv2.legacy.TrackerMOSSE_create not found. Is OpenCV contrib installed? Defaulting to CSRT.")
                return cv2.TrackerCSRT_create()
        elif tracker_type_str_local.upper() == 'MIL': return cv2.TrackerMIL_create()
        # Add other trackers like TLD, MedianFlow, GOTURN if needed and available
        else:
            logger.warning(f"Unknown or unsupported tracker type '{tracker_type_str_local}', defaulting to CSRT.")
            return cv2.TrackerCSRT_create()
    except AttributeError as e:
        logger.error(f"AttributeError creating tracker '{tracker_type_str_local}': {e}. Is OpenCV contrib installed if needed? Defaulting to CSRT.")
        return cv2.TrackerCSRT_create() # Fallback
    except Exception as e:
        logger.error(f"General error creating tracker '{tracker_type_str_local}': {e}. Defaulting to CSRT.")
        return cv2.TrackerCSRT_create() # Fallback


def _load_cnn_bbox_annotations():
    global _cnn_annot_annotations_data
    _cnn_annot_annotations_data = [] # Reset before loading
    if not hasattr(config, 'CNN_ANNOTATIONS_FILE') or not config.CNN_ANNOTATIONS_FILE:
        logger.error("config.CNN_ANNOTATIONS_FILE is not defined.")
        return
    
    annotations_file_path = config.CNN_ANNOTATIONS_FILE
    if os.path.exists(annotations_file_path):
        try:
            with open(annotations_file_path, 'r') as f:
                content = f.read()
            if content.strip(): # Ensure file is not empty
                _cnn_annot_annotations_data = json.loads(content)
                logger.info(f"Loaded {len(_cnn_annot_annotations_data)} CNN bbox annotations from {annotations_file_path}.")
            else:
                logger.info(f"CNN bbox annotation file {annotations_file_path} is empty. Starting fresh.")
        except json.JSONDecodeError:
            logger.warning(f"CNN bbox annotation file {annotations_file_path} is malformed. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading CNN bbox annotations from {annotations_file_path}: {e}")
    else:
        logger.info(f"CNN bbox annotation file {annotations_file_path} not found. Starting fresh.")

def _save_cnn_bbox_annotations():
    global _cnn_annot_annotations_data
    if not hasattr(config, 'CNN_ANNOTATIONS_FILE') or not config.CNN_ANNOTATIONS_FILE:
        logger.error("config.CNN_ANNOTATIONS_FILE is not defined. Cannot save annotations.")
        return

    annotations_file_path = config.CNN_ANNOTATIONS_FILE
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(annotations_file_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
            os.makedirs(output_dir, exist_ok=True)
        
        # Backup existing file
        if os.path.exists(annotations_file_path):
            backup_file = annotations_file_path + ".bak"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(annotations_file_path, backup_file)
            logger.debug(f"Backed up existing annotations to {backup_file}")

        with open(annotations_file_path, 'w') as f:
            json.dump(_cnn_annot_annotations_data, f, indent=4)
        logger.info(f"CNN bbox annotations saved ({len(_cnn_annot_annotations_data)}) to {annotations_file_path}.")
    except Exception as e:
        logger.error(f"Error saving CNN bbox annotations to {annotations_file_path}: {e}")


def on_mouse_draw_roi_cnn(event, x, y, flags, param):
    global _cnn_annot_current_roi_points, _cnn_annot_in_annotation_mode, _cnn_annot_temp_frame_for_annotation
    global _cnn_annot_annotating_id_pending, _cnn_annot_current_video_path_global, _cnn_annot_current_frame_num_global
    global _cnn_annot_annotations_data, _cnn_annot_active_trackers, _cnn_annot_current_bboxes

    if not _cnn_annot_in_annotation_mode or _cnn_annot_temp_frame_for_annotation is None or _cnn_annot_annotating_id_pending is None:
        return

    is_bg_anno = (_cnn_annot_annotating_id_pending == 0)
    target_profile = None
    target_name_label = "Background"
    target_patch_dir = config.BACKGROUND_PATCHES_DIR_CNN
    target_class_idx = config.CNN_BACKGROUND_CLASS_IDX
    target_draw_color = getattr(config, 'BACKGROUND_ANNOTATION_COLOR_CNN', (0, 255, 0)) # Default Green for BG

    if not is_bg_anno: # It's a hippo
        if _cnn_annot_annotating_id_pending in config.HIPPO_PROFILES_CNN:
            target_profile = config.HIPPO_PROFILES_CNN[_cnn_annot_annotating_id_pending]
            target_name_label = target_profile.get("name", f"Hippo {_cnn_annot_annotating_id_pending}")
            target_patch_dir = target_profile.get("patch_dir", config.CNN_PATCHES_DIR) 
            target_class_idx = target_profile.get("class_idx", -1) 
            target_draw_color = target_profile.get("color", (0, 0, 255)) # Default Red for undefined hippo color
        else:
            logger.warning(f"Annotating ID {_cnn_annot_annotating_id_pending} requested, but not found in config.HIPPO_PROFILES_CNN.")
            # Cancel this annotation attempt
            _cnn_annot_annotating_id_pending = None 
            _cnn_annot_current_roi_points = []
            # Attempt to show the original temp frame if it exists, otherwise a blank one
            if _cnn_annot_temp_frame_for_annotation is not None:
                cv2.imshow('CNN BBox Annotation', _cnn_annot_temp_frame_for_annotation.copy())
            return

    # Make a fresh copy of the temp_frame for drawing the current ROI and existing bboxes
    frame_for_roi_drawing = _cnn_annot_temp_frame_for_annotation.copy()
    
    # Draw existing persistent (tracked hippo) bboxes as context
    for hid_existing, bbox_existing in _cnn_annot_current_bboxes.items():
        if bbox_existing: # These are hippo bboxes
            profile_existing = config.HIPPO_PROFILES_CNN.get(hid_existing)
            if profile_existing:
                color_existing = profile_existing.get("color", (255, 0, 0)) # Default Blue
                name_existing = profile_existing.get("name_short", profile_existing.get("name", f"H{hid_existing}"))
                cv2.rectangle(frame_for_roi_drawing, 
                              (bbox_existing[0], bbox_existing[1]), 
                              (bbox_existing[0] + bbox_existing[2], bbox_existing[1] + bbox_existing[3]), 
                              color_existing, 1) # Thinner line for context
                cv2.putText(frame_for_roi_drawing, name_existing, 
                            (bbox_existing[0], bbox_existing[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_existing, 1, cv2.LINE_AA)

    if event == cv2.EVENT_LBUTTONDOWN:
        _cnn_annot_current_roi_points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and len(_cnn_annot_current_roi_points) == 1:
        # Draw the ROI being created with its specific color
        cv2.rectangle(frame_for_roi_drawing, _cnn_annot_current_roi_points[0], (x, y), target_draw_color, 2)
        cv2.imshow('CNN BBox Annotation', frame_for_roi_drawing)

    elif event == cv2.EVENT_LBUTTONUP and len(_cnn_annot_current_roi_points) == 1:
        _cnn_annot_current_roi_points.append((x, y))
        x1, y1 = _cnn_annot_current_roi_points[0]
        x2, y2 = _cnn_annot_current_roi_points[1]
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        roi_x = min(x1, x2)
        roi_y = min(y1, y2)
        roi_w = abs(x1 - x2)
        roi_h = abs(y1 - y2)
        bbox = (roi_x, roi_y, roi_w, roi_h)

        if bbox[2] > 5 and bbox[3] > 5: # ROI is valid size
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            video_basename = os.path.basename(_cnn_annot_current_video_path_global)
            safe_video_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(video_basename)[0])
            
            patch_file_prefix = "bg_annot_" if is_bg_anno else f"hippo{_cnn_annot_annotating_id_pending}_"
            patch_filename = f"{patch_file_prefix}{safe_video_name}_f{_cnn_annot_current_frame_num_global}_{timestamp_str}.png"
            
            os.makedirs(target_patch_dir, exist_ok=True)
            patch_full_path = os.path.join(target_patch_dir, patch_filename)
            
            # Extract patch from the original _cnn_annot_temp_frame_for_annotation (clean frame)
            patch_image = _cnn_annot_temp_frame_for_annotation[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

            if patch_image.size == 0:
                logger.warning("Attempted to save an empty patch! ROI likely out of bounds or invalid.")
                _cnn_annot_current_roi_points = [] # Reset ROI points
                 # Redraw frame_for_roi_drawing which has context, but no new ROI
                cv2.imshow('CNN BBox Annotation', frame_for_roi_drawing)
                return

            try:
                cv2.imwrite(patch_full_path, patch_image)
                annotation_entry = {
                    "video_path": _cnn_annot_current_video_path_global,
                    "frame_number": _cnn_annot_current_frame_num_global,
                    "label_id_profile": _cnn_annot_annotating_id_pending,
                    "class_idx_cnn": target_class_idx,
                    "bbox": list(bbox),
                    "patch_path": patch_full_path
                }
                _cnn_annot_annotations_data.append(annotation_entry)
                logger.info(f"Annotated '{target_name_label}', BBox: {bbox}. Patch saved: {patch_full_path}")
                _save_cnn_bbox_annotations()

                if not is_bg_anno and target_profile: # It's a hippo, so initialize/update tracker
                    hippo_id_to_track = _cnn_annot_annotating_id_pending
                    try:
                        tracker = create_cv_tracker_cnn()
                        # Initialize tracker on the same frame the ROI was defined on
                        tracker.init(_cnn_annot_temp_frame_for_annotation, bbox)
                        _cnn_annot_active_trackers[hippo_id_to_track] = tracker
                        _cnn_annot_current_bboxes[hippo_id_to_track] = bbox # This makes it persistent for display
                        logger.info(f"Tracker for '{target_name_label}' (ID: {hippo_id_to_track}) initialized at {bbox}")
                    except Exception as e_track:
                        logger.error(f"Error initializing tracker for '{target_name_label}': {e_track}")
                        if hippo_id_to_track in _cnn_annot_active_trackers:
                            _cnn_annot_active_trackers[hippo_id_to_track] = None
                        if hippo_id_to_track in _cnn_annot_current_bboxes:
                             _cnn_annot_current_bboxes[hippo_id_to_track] = None
            except Exception as e_save:
                logger.error(f"Error during patch saving or annotation appending for '{target_name_label}': {e_save}")
            
            # Reset for next potential annotation, back to ID selection phase
            _cnn_annot_annotating_id_pending = None
            _cnn_annot_current_roi_points = []
            
            # Show the frame_for_roi_drawing which now includes the just-drawn successful ROI (if BG)
            # or it will be drawn by the main loop's _cnn_annot_current_bboxes logic (if Hippo)
            if is_bg_anno: # Draw the BG ROI one last time on this static frame display
                 cv2.rectangle(frame_for_roi_drawing, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), target_draw_color, 2)
            cv2.imshow('CNN BBox Annotation', frame_for_roi_drawing)

        else: # ROI too small or invalid
            logger.info("ROI too small (w or h <= 5px). Drawing cancelled.")
            _cnn_annot_current_roi_points = []
            cv2.imshow('CNN BBox Annotation', frame_for_roi_drawing) # Show context without the invalid ROI attempt


def run_cnn_bbox_annotation_mode():
    global _cnn_annot_current_video_path_global, _cnn_annot_current_frame_num_global
    global _cnn_annot_in_annotation_mode, _cnn_annot_annotating_id_pending
    global _cnn_annot_temp_frame_for_annotation, _cnn_annot_current_roi_points
    global _cnn_annot_active_trackers, _cnn_annot_current_bboxes

    _load_cnn_bbox_annotations()
    _initialize_kernels_cnn()

    cv2.namedWindow('CNN BBox Annotation')
    cv2.setMouseCallback('CNN BBox Annotation', on_mouse_draw_roi_cnn)

    video_source_dir = config.CLIPS_DIR # Assuming clips are in CLIPS_DIR
    if not os.path.isdir(video_source_dir):
        logger.error(f"Source directory for CNN annotation clips not found: {video_source_dir}")
        return

    video_files = sorted([
        os.path.join(video_source_dir, f)
        for f in os.listdir(video_source_dir)
        if f.lower().endswith(getattr(config, 'VIDEO_EXTENSIONS', ('.mp4', '.avi', '.mov')))
    ])

    if not video_files:
        logger.error(f"No video clips found in '{video_source_dir}'. Ensure clips are generated.")
        return

    logger.info("--- CNN Bounding Box Annotation Mode ---")
    # Initial instructions will be updated dynamically in the loop

    user_wants_to_quit_all = False
    for video_idx, vid_path in enumerate(video_files):
        if user_wants_to_quit_all:
            break

        _cnn_annot_current_video_path_global = vid_path
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {vid_path}")
            continue

        logger.info(f"\nAnnotating Video {video_idx + 1}/{len(video_files)}: {os.path.basename(vid_path)}")

        _cnn_annot_current_frame_num_global = 0
        paused = False
        # _cnn_annot_in_annotation_mode persists across videos unless toggled by user
        _cnn_annot_annotating_id_pending = None # Reset for new video
        _cnn_annot_current_roi_points = []      # Reset for new video
        _cnn_annot_temp_frame_for_annotation = None # Reset for new video
        
        # Reset trackers and current bboxes for each new video
        _cnn_annot_active_trackers = {hid: None for hid in config.HIPPO_PROFILES_CNN.keys()}
        _cnn_annot_current_bboxes = {hid: None for hid in config.HIPPO_PROFILES_CNN.keys()}
        
        latest_raw_frame = None # Stores the most recent raw frame from cap.read()

        while True:
            if not paused:
                ret_frame, frame_data = cap.read()
                if not ret_frame:
                    break # End of video or error
                _cnn_annot_current_frame_num_global += 1
                latest_raw_frame = frame_data.copy() # Keep a clean copy
            
            if latest_raw_frame is None: # Should only happen if video is empty or first frame fails
                logger.warning(f"No frame could be read from {vid_path}. Skipping to next video.")
                break

            # Frame to draw UI elements and current state ON for display
            display_frame_with_ui = latest_raw_frame.copy() 

            # Update active trackers if not in the middle of drawing a new ROI
            # (i.e., LBUTTONDOWN has occurred but not LBUTTONUP yet for the current ROI being drawn)
            if not (_cnn_annot_in_annotation_mode and _cnn_annot_annotating_id_pending is not None and len(_cnn_annot_current_roi_points) == 1):
                for hid_tracker, tracker_instance in list(_cnn_annot_active_trackers.items()): # Use list for safe modification
                    if tracker_instance is not None:
                        success, bbox_cv = tracker_instance.update(latest_raw_frame) # Update on clean current frame
                        if success:
                            _cnn_annot_current_bboxes[hid_tracker] = tuple(map(int, bbox_cv))
                        else: # Tracker lost target
                            logger.debug(f"Tracker lost for Hippo ID {hid_tracker} on frame {_cnn_annot_current_frame_num_global}")
                            _cnn_annot_active_trackers[hid_tracker] = None # Remove tracker
                            _cnn_annot_current_bboxes[hid_tracker] = None # Remove its bbox from display
            
            # Draw persistent (tracked hippo) bboxes on the display_frame_with_ui
            for hid_draw, bbox_draw in _cnn_annot_current_bboxes.items():
                if bbox_draw: # These are hippo bboxes
                    profile_draw = config.HIPPO_PROFILES_CNN.get(hid_draw)
                    if profile_draw:
                        color = profile_draw.get("color", (255, 0, 0)) # Default Blue
                        name = profile_draw.get("name_short", profile_draw.get("name", f"H{hid_draw}"))
                        cv2.rectangle(display_frame_with_ui, (bbox_draw[0], bbox_draw[1]),
                                      (bbox_draw[0] + bbox_draw[2], bbox_draw[1] + bbox_draw[3]), color, 2)
                        cv2.putText(display_frame_with_ui, name, (bbox_draw[0], bbox_draw[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # --- Information Text (Dynamic) ---
            info_lines = []
            info_lines.append(f"Vid:{video_idx+1}/{len(video_files)} Fr:{_cnn_annot_current_frame_num_global} File: {os.path.basename(vid_path)}")
            
            status_line_parts = []
            if paused: status_line_parts.append("PAUSED")
            
            if _cnn_annot_in_annotation_mode:
                status_line_parts.append("AnnoMode:ON")
                if _cnn_annot_annotating_id_pending is not None: # User is currently expected to draw for this ID
                    entity_name = "Background" if _cnn_annot_annotating_id_pending == 0 else \
                                  config.HIPPO_PROFILES_CNN.get(_cnn_annot_annotating_id_pending, {}).get("name", f"ID {_cnn_annot_annotating_id_pending}")
                    status_line_parts.append(f"DRAW ROI for: {entity_name} | [ESC]CancelDraw")
                else: # User needs to select an ID to annotate
                    id_options_str = "[0]BG " + " ".join([f"[{hid}]{prof.get('name_short',f'H{hid}')}" 
                                                          for hid, prof in sorted(config.HIPPO_PROFILES_CNN.items())])
                    status_line_parts.append(f"SELECT ID: {id_options_str} | ([a]ModeOFF)")
            else: # Annotation mode is OFF
                status_line_parts.append("AnnoMode:OFF ([a]ModeON)")
            
            info_lines.append(" | ".join(status_line_parts))
            info_lines.append("Keys: [p]Pause [f]NxtFr [s]SaveAll [n]NxtVid [q]QuitAll")

            y_offset = 20
            for line_idx, line_text in enumerate(info_lines):
                cv2.putText(display_frame_with_ui, line_text, (10, y_offset + line_idx * 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 255), 1, cv2.LINE_AA) # Yellow text
            
            cv2.imshow('CNN BBox Annotation', display_frame_with_ui)

            # --- Key Handling ---
            key_wait_time = 0 if (paused or (_cnn_annot_in_annotation_mode and _cnn_annot_annotating_id_pending is not None)) else 30 
            key = cv2.waitKey(key_wait_time) & 0xFF

            if key == ord('q'):
                user_wants_to_quit_all = True; break
            elif key == ord('n'):
                break # Go to next video
            elif key == ord('p'):
                paused = not paused
                logger.info("Playback Paused" if paused else "Playback Resumed")
            elif key == ord('f') and paused: # Advance one frame
                ret_adv, frame_adv = cap.read()
                if ret_adv:
                    _cnn_annot_current_frame_num_global += 1
                    latest_raw_frame = frame_adv.copy()
                    if _cnn_annot_in_annotation_mode and _cnn_annot_annotating_id_pending is not None:
                        _cnn_annot_temp_frame_for_annotation = latest_raw_frame.copy()
                        _cnn_annot_current_roi_points = [] # Reset points for new frame
                        logger.info(f"Advanced to frame {_cnn_annot_current_frame_num_global}. Redraw ROI for selected ID if needed.")
                else: # End of video while advancing
                    logger.info("End of video reached while advancing frame.")
                    break 
            elif key == ord('s'):
                _save_cnn_bbox_annotations() # Explicit save
            elif key == ord('a'): # Toggle annotation mode
                _cnn_annot_in_annotation_mode = not _cnn_annot_in_annotation_mode
                if not _cnn_annot_in_annotation_mode: # If turning OFF annotation mode
                    _cnn_annot_annotating_id_pending = None # Cancel any pending drawing selection
                    _cnn_annot_current_roi_points = []
                    _cnn_annot_temp_frame_for_annotation = None
                logger.info(f"Annotation Mode: {'ON' if _cnn_annot_in_annotation_mode else 'OFF'}")
            
            elif _cnn_annot_in_annotation_mode:
                if _cnn_annot_annotating_id_pending is None: # If ready to select an ID
                    selected_id_to_annotate = None
                    if ord('0') <= key <= ord('9'): 
                        try:
                            key_as_int = int(chr(key))
                            if key_as_int == 0: 
                                selected_id_to_annotate = 0
                            elif key_as_int in config.HIPPO_PROFILES_CNN: 
                                selected_id_to_annotate = key_as_int
                            else:
                                logger.warning(f"Hippo ID {key_as_int} selected via key '{chr(key)}' is not defined in config.HIPPO_PROFILES_CNN.")
                        except ValueError: pass 

                    if selected_id_to_annotate is not None:
                        _cnn_annot_annotating_id_pending = selected_id_to_annotate
                        _cnn_annot_current_roi_points = [] 
                        _cnn_annot_temp_frame_for_annotation = latest_raw_frame.copy() 
                        
                        entity_name = "Background" if _cnn_annot_annotating_id_pending == 0 else \
                                      config.HIPPO_PROFILES_CNN.get(_cnn_annot_annotating_id_pending, {}).get("name", f"ID {_cnn_annot_annotating_id_pending}")
                        logger.info(f"Selected '{entity_name}' (ID: {_cnn_annot_annotating_id_pending}) for annotation. Draw ROI with mouse.")
                        paused = True 

                elif _cnn_annot_annotating_id_pending is not None and key == 27: # ESC key to cancel current drawing selection
                    logger.info(f"Cancelled drawing for ID: {_cnn_annot_annotating_id_pending}.")
                    _cnn_annot_annotating_id_pending = None
                    _cnn_annot_current_roi_points = []
                    _cnn_annot_temp_frame_for_annotation = None
        
        cap.release()
        _save_cnn_bbox_annotations() 

    if cv2.getWindowProperty('CNN BBox Annotation', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow('CNN BBox Annotation')
    logger.info("Exited CNN BBox Annotation Mode.")


def run_motion_detector_cnn(frame_local, prev_gray_local, back_sub_local):
    _initialize_kernels_cnn() # Ensure kernels are ready
    candidate_proposals = []
    # Gaussian blur might be too aggressive or might need adjustment based on hippo size/distance
    processed_frame = cv2.GaussianBlur(frame_local, config.GAUSSIAN_BLUR_KERNEL_SIZE_CNN, 0)
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    # Background subtraction
    fg_mask_raw = back_sub_local.apply(processed_frame, learningRate=config.MOG2_LEARNING_RATE_CNN) # Use configured learning rate
    # Morphological operations for BG subtraction mask
    fg_mask_bg_sub = cv2.morphologyEx(fg_mask_raw, cv2.MORPH_OPEN, open_kernel_bg_cnn)
    fg_mask_bg_sub = cv2.morphologyEx(fg_mask_bg_sub, cv2.MORPH_CLOSE, close_kernel_bg_cnn)
    contours_bg, _ = cv2.findContours(fg_mask_bg_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours_info = [{'c': c, 's': 'bg_sub'} for c in contours_bg] # Store source
    
    # Frame differencing
    if prev_gray_local is not None:
        frame_diff = cv2.absdiff(prev_gray_local, gray_frame)
        _, thresh_diff = cv2.threshold(frame_diff, config.FRAME_DIFF_THRESHOLD_CNN, 255, cv2.THRESH_BINARY)
        # Dilate might be better than open for frame diff, to connect broken parts of moving objects
        dilated_diff = cv2.dilate(thresh_diff, open_kernel_fd_cnn, iterations=config.FRAME_DIFF_DILATE_ITER_CNN) 
        fg_mask_frame_diff = cv2.morphologyEx(dilated_diff, cv2.MORPH_CLOSE, close_kernel_fd_cnn)
        contours_fd, _ = cv2.findContours(fg_mask_frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_fd:
            all_contours_info.append({'c': c, 's': 'frame_diff'})
            
    for item in all_contours_info:
        contour = item['c']
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        # Filter based on area and aspect ratio if needed
        if area < config.MIN_CONTOUR_AREA_CNN: # Use a config value for min area
            continue
        # Optionally, add aspect ratio filtering: aspect_ratio = w / float(h) ...
        candidate_proposals.append({'contour':contour,'bbox':(x,y,w,h),'area':area,'source':item['s']})
        
    return candidate_proposals, gray_frame


def run_cnn_on_proposals(frame_local, motion_proposals_local, cnn_model_instance):
    """Runs the provided CNN model on image patches from motion proposals."""
    if cnn_model_instance is None:
        logger.debug("CNN model instance not provided to run_cnn_on_proposals. Skipping.")
        return {hid: [] for hid in config.HIPPO_PROFILES_CNN.keys()} # Return empty dict matching expected structure

    cnn_detections = {hid: [] for hid in config.HIPPO_PROFILES_CNN.keys()} # Initialize for all profiles
    patches_for_cnn_batch, original_proposal_info_batch = [], []

    for prop in motion_proposals_local:
        x,y,w,h = prop['bbox']
        if w < config.MIN_PATCH_WIDTH_CNN or h < config.MIN_PATCH_HEIGHT_CNN: # Use config values
            continue
        patch = frame_local[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        
        resized_patch = cv2.resize(patch, (config.CNN_IMG_WIDTH, config.CNN_IMG_HEIGHT))
        normalized_patch = resized_patch.astype("float32") / 255.0 # Normalization
        patches_for_cnn_batch.append(normalized_patch)
        original_proposal_info_batch.append(prop)

    if not patches_for_cnn_batch:
        return cnn_detections 

    patches_array = np.array(patches_for_cnn_batch)
    # Ensure patches_array is 4D (batch_size, height, width, channels)
    if patches_array.ndim == 3 and len(patches_for_cnn_batch) == 1: # Single patch
        patches_array = np.expand_dims(patches_array, axis=0)
    elif patches_array.ndim != 4 and len(patches_for_cnn_batch) > 1: # Multiple patches but wrong dimensions
        logger.warning(f"Patch array has unexpected dimensions: {patches_array.shape}. Expected 4D. Skipping CNN prediction for this batch.")
        return cnn_detections
    
    if patches_array.shape[0] == 0: # Should be caught by "if not patches_for_cnn_batch"
        return cnn_detections
    
    try:
        predictions = cnn_model_instance.predict(patches_array, verbose=0) # Use passed model instance
    except Exception as e:
        logger.error(f"Error during CNN model prediction: {e}", exc_info=True)
        return cnn_detections


    for i, scores in enumerate(predictions):
        original_prop = original_proposal_info_batch[i]
        predicted_class_idx = np.argmax(scores)
        confidence = scores[predicted_class_idx]

        if confidence >= config.CNN_CONFIDENCE_THRESHOLD and predicted_class_idx != config.CNN_BACKGROUND_CLASS_IDX:
            target_hippo_id = None
            # Find which hippo profile this predicted_class_idx corresponds to
            for hid_profile, profile_data in config.HIPPO_PROFILES_CNN.items():
                if profile_data.get("class_idx") == predicted_class_idx:
                    target_hippo_id = hid_profile
                    break
            
            if target_hippo_id is not None:
                cnn_detections[target_hippo_id].append({
                    'bbox': original_prop['bbox'], 
                    'confidence': float(confidence), 
                    'source': 'cnn_' + original_prop['source'] # e.g., cnn_bg_sub or cnn_frame_diff
                })
    return cnn_detections

class CustomCNNDetector:
    def __init__(self):
        self.bg_subtractor = None
        self.prev_gray_frame = None
        self.loaded_cnn_model = None # Instance variable for the Keras model
        _initialize_kernels_cnn() # Initialize morphological kernels once
        self._load_model_instance() # Attempt to load the model into self.loaded_cnn_model

    def _load_model_instance(self):
        """Loads the trained Keras CNN model into self.loaded_cnn_model."""
        if not hasattr(config, 'CNN_MODEL_SAVE_PATH') or not config.CNN_MODEL_SAVE_PATH:
            logger.error("config.CNN_MODEL_SAVE_PATH is not defined.")
            self.loaded_cnn_model = None
            return False
        
        model_path = config.CNN_MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            logger.warning(f"Trained CNN model not found at {model_path} for CustomCNNDetector instance.")
            self.loaded_cnn_model = None
            return False
        try:
            self.loaded_cnn_model = load_model(model_path)
            logger.info(f"CustomCNNDetector: Keras CNN model loaded from: {model_path}")
            return True
        except Exception as e:
            logger.error(f"CustomCNNDetector: Error loading Keras CNN model: {e}", exc_info=True)
            self.loaded_cnn_model = None
            return False

    def process_segment_for_detections(self, segment_info):
        if self.loaded_cnn_model is None:
            logger.error("CNN model not loaded in CustomCNNDetector instance. Cannot process segment for detections.")
            if not self._load_model_instance(): # Attempt to load again
                 logger.error("Failed to load CNN model on demand. Aborting detection.")
                 return []


        original_video_path = segment_info['video_path']
        # Construct clip path from segment_info (assuming it might be different from original_video_path)
        # This depends on how segment_info['clip_path_for_processing'] or similar is provided.
        # For now, let's assume segment_info provides the path to the clip to be processed.
        # If clips are always in config.CLIPS_DIR and named based on segment_id:
        clip_basename_from_original = os.path.basename(original_video_path)
        segment_id = f"{os.path.splitext(clip_basename_from_original)[0]}_segment_{segment_info['start_frame']}_{segment_info['end_frame']}"
        clip_to_process_path = os.path.join(config.CLIPS_DIR, f"{segment_id}.mp4")


        if not os.path.exists(clip_to_process_path):
            logger.warning(f"Clip not found for segment: {clip_to_process_path}. Cannot process with CNN detector.")
            return []

        cap = cv2.VideoCapture(clip_to_process_path)
        if not cap.isOpened():
            logger.error(f"Could not open clip for CNN detection: {clip_to_process_path}")
            return []

        logger.info(f"CNN Detector processing clip: {os.path.basename(clip_to_process_path)}")
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY_CNN, 
            varThreshold=config.MOG2_VAR_THRESHOLD_CNN,
            detectShadows=config.MOG2_DETECT_SHADOWS_CNN
        )
        self.prev_gray_frame = None
        
        clip_trackers = {hid: None for hid in config.HIPPO_PROFILES_CNN.keys()}
        clip_current_bboxes = {hid: None for hid in config.HIPPO_PROFILES_CNN.keys()}

        segment_detections_data = []
        frame_idx_in_clip = 0 # Relative to the start of the clip_to_process_path

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            motion_proposals, current_gray = run_motion_detector_cnn(frame, self.prev_gray_frame, self.bg_subtractor)
            self.prev_gray_frame = current_gray
            
            cnn_detections_for_frame = run_cnn_on_proposals(frame, motion_proposals, self.loaded_cnn_model)

            for hid_profile_id in config.HIPPO_PROFILES_CNN.keys():
                # Try to update from tracker first
                if clip_trackers.get(hid_profile_id) is not None:
                    success_track, bbox_cv_track = clip_trackers[hid_profile_id].update(frame)
                    if success_track:
                        clip_current_bboxes[hid_profile_id] = tuple(map(int, bbox_cv_track))
                    else: 
                        clip_trackers[hid_profile_id] = None
                        clip_current_bboxes[hid_profile_id] = None 
                
                # If CNN has a (better) detection, prioritize it and re-init tracker
                if cnn_detections_for_frame.get(hid_profile_id) and cnn_detections_for_frame[hid_profile_id]:
                    best_cnn_detection = max(cnn_detections_for_frame[hid_profile_id], key=lambda d: d['confidence'])
                    # Simple rule: if CNN conf is high, or if tracker lost, or if CNN bbox is "better"
                    # For now, let's just use CNN if confidence is good.
                    if best_cnn_detection['confidence'] > getattr(config, 'CNN_DETECTION_OVERWRITE_TRACKER_CONF_THRESHOLD', 0.6):
                        clip_current_bboxes[hid_profile_id] = best_cnn_detection['bbox'] 
                        try:
                            new_tracker = create_cv_tracker_cnn()
                            new_tracker.init(frame, best_cnn_detection['bbox'])
                            clip_trackers[hid_profile_id] = new_tracker
                        except Exception as e_trk_reinit:
                            logger.debug(f"Could not re-init tracker for hippo {hid_profile_id} from CNN det: {e_trk_reinit}")
                            clip_trackers[hid_profile_id] = None 

            output_bboxes_for_this_frame_with_conf = []
            for hid_prof_id_sorted in sorted(config.HIPPO_PROFILES_CNN.keys()): 
                current_bbox_for_hippo = clip_current_bboxes.get(hid_prof_id_sorted)
                current_conf = 0.5 # Default/placeholder if only from tracker
                
                if current_bbox_for_hippo:
                    # Try to find if this current_bbox_for_hippo came from a direct CNN detection this frame
                    # to get its original confidence.
                    is_from_cnn_this_frame = False
                    if cnn_detections_for_frame.get(hid_prof_id_sorted):
                        for cnn_det in cnn_detections_for_frame[hid_prof_id_sorted]:
                            if tuple(cnn_det['bbox']) == current_bbox_for_hippo: # Simple bbox match
                                current_conf = cnn_det['confidence']
                                is_from_cnn_this_frame = True
                                break
                    
                    output_bboxes_for_this_frame_with_conf.append(
                        list(current_bbox_for_hippo) + [current_conf]
                    )
                else:
                    output_bboxes_for_this_frame_with_conf.append([np.nan] * config.BBOX_FEATURE_LENGTH) # Match BBOX_FEATURE_LENGTH

            # Absolute frame index in the original video
            original_frame_idx_abs = segment_info['start_frame'] + frame_idx_in_clip
            timestamp = original_frame_idx_abs / segment_info['fps'] if segment_info.get('fps', 0) > 0 else 0.0

            segment_detections_data.append({
                'video_path': original_video_path, 
                'frame_idx': original_frame_idx_abs,
                'timestamp_sec': timestamp,
                'hippo_bboxes_conf': output_bboxes_for_this_frame_with_conf # List of [x,y,w,h,conf]
            })
            frame_idx_in_clip += 1
        
        cap.release()
        logger.info(f"CNN Detector finished processing clip: {os.path.basename(clip_to_process_path)}. Output {len(segment_detections_data)} frames data.")
        return segment_detections_data


def create_lightweight_cnn_architecture(input_shape=None, num_classes=None):
    if input_shape is None:
        input_shape = (config.CNN_IMG_HEIGHT, config.CNN_IMG_WIDTH, 3)
    if num_classes is None:
        num_classes = config.CNN_NUM_CLASSES

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer_choice = getattr(config, 'CNN_OPTIMIZER', 'Adam')
    learning_rate = getattr(config, 'CNN_LEARNING_RATE', 0.0005)

    if optimizer_choice.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    # Add other optimizers if needed, e.g., AdamW, SGD
    # elif optimizer_choice.lower() == 'adamw':
    #     from tensorflow.keras.optimizers.experimental import AdamW # May need tf-nightly or specific TF version
    #     optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.004)
    else:
        logger.warning(f"Unsupported optimizer '{optimizer_choice}' in config. Defaulting to Adam.")
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("Lightweight CNN architecture created and compiled.")
    model.summary(print_fn=logger.info)
    return model


def train_custom_cnn_detector_model():
    global module_loaded_cnn_model # Ensure this refers to the module-level variable
    logger.info("\n--- Custom CNN Detector Training ---")
    
    _load_cnn_bbox_annotations() # Loads into _cnn_annot_annotations_data
    images, labels = [], []
    loaded_patch_paths_normalized = set() # To avoid duplicates if BG patches are also in annotations

    if not _cnn_annot_annotations_data:
        logger.error("No CNN bbox annotations found. Cannot train model.")
        return

    for ann in _cnn_annot_annotations_data:
        patch_path = ann.get("patch_path")
        class_idx = ann.get("class_idx_cnn")
        if not patch_path or class_idx is None:
            logger.warning(f"Skipping annotation due to missing patch_path or class_idx_cnn: {ann}")
            continue
        
        if not os.path.exists(patch_path):
            logger.warning(f"Patch file missing, skipping: {patch_path}")
            continue
        
        img = cv2.imread(patch_path)
        if img is not None:
            img_resized = cv2.resize(img, (config.CNN_IMG_WIDTH, config.CNN_IMG_HEIGHT))
            images.append(img_resized)
            labels.append(class_idx)
            loaded_patch_paths_normalized.add(os.path.normpath(patch_path))
        else:
            logger.warning(f"Could not read image from patch: {patch_path}")

    # Load additional general background patches
    general_bg_patches_loaded_count = 0
    if hasattr(config, 'BACKGROUND_PATCHES_DIR_CNN') and os.path.exists(config.BACKGROUND_PATCHES_DIR_CNN):
        for img_name in os.listdir(config.BACKGROUND_PATCHES_DIR_CNN):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                general_bg_patch_path = os.path.join(config.BACKGROUND_PATCHES_DIR_CNN, img_name)
                # Avoid loading if it was already loaded via an annotation entry
                if os.path.normpath(general_bg_patch_path) in loaded_patch_paths_normalized:
                    continue 
                
                img_bg = cv2.imread(general_bg_patch_path)
                if img_bg is not None:
                    img_bg_resized = cv2.resize(img_bg, (config.CNN_IMG_WIDTH, config.CNN_IMG_HEIGHT))
                    images.append(img_bg_resized)
                    labels.append(config.CNN_BACKGROUND_CLASS_IDX) # Assume these are all background
                    general_bg_patches_loaded_count += 1
    logger.info(f"Loaded {general_bg_patches_loaded_count} additional general background patches.")

    if not images:
        logger.error("No images loaded for training after processing annotations and BG patches.")
        return

    # Log class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution in training data:")
    for lbl, count in zip(unique_labels, counts):
        if lbl == config.CNN_BACKGROUND_CLASS_IDX:
            logger.info(f"  Background (Class {lbl}): {count} samples")
        else:
            profile_name = "Unknown Hippo Class"
            for hid, prof_data in config.HIPPO_PROFILES_CNN.items():
                if prof_data.get("class_idx") == lbl:
                    profile_name = prof_data.get("name", f"Hippo Profile (ID {hid})")
                    break
            logger.info(f"  {profile_name} (Class {lbl}): {count} samples")
    
    if len(unique_labels) < config.CNN_NUM_CLASSES:
         logger.warning(f"Warning: Only {len(unique_labels)} unique classes found, but CNN_NUM_CLASSES is {config.CNN_NUM_CLASSES}. "
                       "Model might not learn to distinguish all configured classes if some have no samples.")
    if len(images) < config.CNN_BATCH_SIZE:
        logger.warning(f"Total samples ({len(images)}) is less than batch size ({config.CNN_BATCH_SIZE}). Consider reducing batch size or adding more data.")

    images_np = np.array(images, dtype="float32") / 255.0
    labels_np = to_categorical(np.array(labels), num_classes=config.CNN_NUM_CLASSES)

    if images_np.shape[0] == 0:
        logger.error("Numpy array for images is empty. Training aborted.")
        return

    # Stratified split if possible
    # Stratification requires at least 2 samples per class for the smallest group in train/test.
    # Simpler check: at least 2 unique classes, and each class appears at least twice.
    can_stratify = len(unique_labels) > 1 and all(c >= 2 for c in counts)
    stratify_target = labels_np if can_stratify else None # Pass the labels for stratification
    if not can_stratify:
        logger.warning("Cannot perform stratified split due to insufficient samples per class or too few classes. Using non-stratified split.")

    train_x, test_x, train_y, test_y = train_test_split(
        images_np, labels_np, 
        test_size=getattr(config, 'CNN_TEST_SPLIT_SIZE', 0.20), 
        stratify=stratify_target, 
        random_state=getattr(config, 'RANDOM_STATE_SEED', 42)
    )

    if train_x.shape[0] == 0:
        logger.error("No training samples after train-test split. Aborting training.")
        return
    logger.info(f"Training set: X_shape={train_x.shape}, Y_shape={train_y.shape}")
    if test_x.shape[0] > 0:
        logger.info(f"Test set: X_shape={test_x.shape}, Y_shape={test_y.shape}")
    else:
        logger.warning("Test set is empty after split. No validation will be performed during training.")


    # Image Data Generator for augmentation
    img_gen = ImageDataGenerator(
        rotation_range=getattr(config, 'AUG_ROTATION_RANGE', 15),
        zoom_range=getattr(config, 'AUG_ZOOM_RANGE', 0.1),
        width_shift_range=getattr(config, 'AUG_WIDTH_SHIFT_RANGE', 0.1),
        height_shift_range=getattr(config, 'AUG_HEIGHT_SHIFT_RANGE', 0.1),
        horizontal_flip=getattr(config, 'AUG_HORIZONTAL_FLIP', True),
        fill_mode=getattr(config, 'AUG_FILL_MODE', "nearest")
    )

    model_cnn_instance = create_lightweight_cnn_architecture() # Get a fresh model instance
    logger.info("Starting CNN model training...")
    
    validation_data_tuple = (test_x, test_y) if test_x.shape[0] > 0 and test_y.shape[0] > 0 else None
    
    # Workers and multiprocessing for fit_generator (img_gen.flow)
    # os.cpu_count() can be None. Max 1 worker on Windows for stability with some Keras/TF versions.
    num_fit_workers = 1 
    use_fit_multiprocessing = False
    if os.name != 'nt': # Multiprocessing often more stable on Linux/macOS
        cpu_cores = os.cpu_count()
        if cpu_cores and cpu_cores > 1:
            num_fit_workers = max(1, cpu_cores // 2 -1) if cpu_cores > 2 else 1
            if num_fit_workers > 1:
                use_fit_multiprocessing = True
    
    logger.info(f"Using {num_fit_workers} workers and multiprocessing={use_fit_multiprocessing} for model.fit")

    try:
        history = model_cnn_instance.fit(
            img_gen.flow(train_x, train_y, batch_size=config.CNN_BATCH_SIZE),
            validation_data=validation_data_tuple,
            epochs=config.CNN_EPOCHS,
            verbose=1, # Or 2 for less output per epoch
            workers=num_fit_workers,
            use_multiprocessing=use_fit_multiprocessing
        )
        logger.info("CNN model training completed.")
        # Optionally, log training history (accuracy, loss)
        # for key in history.history.keys():
        #     logger.debug(f"Training history for {key}: {history.history[key]}")

        model_cnn_instance.save(config.CNN_MODEL_SAVE_PATH)
        logger.info(f"Trained CNN model saved to: {config.CNN_MODEL_SAVE_PATH}")
        module_loaded_cnn_model = model_cnn_instance # Update module-level model with the newly trained one
    
    except Exception as e:
        logger.error(f"An error occurred during CNN model training: {e}", exc_info=True)


def infer_behavior_and_emotion_cnn(hippo_profiles_current_state, frame_area_px=None):
    """
    Infers basic behavior and emotion based on hippo state (bbox, prev_bbox).
    This is a placeholder/example inference logic.
    Args:
        hippo_profiles_current_state (dict): {hippo_id: {'bbox': (x,y,w,h), 'prev_bbox': (x,y,w,h), ...}, ...}
        frame_area_px (float, optional): Total area of the video frame in pixels. For relative size.
    Returns:
        tuple: (behaviors_dict, emotions_dict)
               behaviors_dict = {hippo_id: "behavior_label", ...}
               emotions_dict = {hippo_id: "emotion_label", ...}
    """
    behaviors = {hid: "out_of_frame" for hid in hippo_profiles_current_state.keys()}
    emotions = {hid: "Neutral" for hid in hippo_profiles_current_state.keys()}
    
    _velocities_normalized = {hid: 0.0 for hid in hippo_profiles_current_state.keys()} # Velocity normalized by avg dimension
    active_hippos_data_for_interaction = {} # Store data for hippos currently in frame

    for hid, profile_state in hippo_profiles_current_state.items():
        current_bbox = profile_state.get("bbox")
        prev_bbox = profile_state.get("prev_bbox")

        if current_bbox and all(not np.isnan(v) for v in current_bbox): # Check for NaN
            active_hippos_data_for_interaction[hid] = current_bbox # Store for interaction check
            
            if prev_bbox and all(not np.isnan(v) for v in prev_bbox):
                c_cx, c_cy = current_bbox[0] + current_bbox[2] / 2.0, current_bbox[1] + current_bbox[3] / 2.0
                p_cx, p_cy = prev_bbox[0] + prev_bbox[2] / 2.0, prev_bbox[1] + prev_bbox[3] / 2.0
                pixel_distance = np.sqrt((c_cx - p_cx)**2 + (c_cy - p_cy)**2)
                
                # Normalize velocity by average dimension of the hippo's bbox
                avg_dimension = (current_bbox[2] + current_bbox[3]) / 2.0
                _velocities_normalized[hid] = pixel_distance / (avg_dimension + 1e-6) # Add epsilon to avoid div by zero
            else:
                _velocities_normalized[hid] = 0.05 # Default small velocity if no previous bbox (e.g., first detection)

            # Basic behavior based on normalized velocity
            vel_norm = _velocities_normalized[hid]
            if vel_norm < getattr(config, 'VEL_THRESH_RESTING_CNN', 0.02):
                behaviors[hid] = "resting"
            elif vel_norm < getattr(config, 'VEL_THRESH_WALKING_CNN', 0.15):
                behaviors[hid] = "walking"
            else: # vel_norm >= 0.15
                behaviors[hid] = "active_general"

            # Basic emotion inference linked to behavior
            if behaviors[hid] == "resting":
                emotions[hid] = "Calm"
            elif behaviors[hid] == "active_general" and vel_norm > getattr(config, 'VEL_THRESH_PLAYFUL_CNN', 0.3):
                 # High activity could be playful or agitated, needs more context
                emotions[hid] = "Playful" # Simplified assumption
            else:
                emotions[hid] = "Neutral"
        else: # Not in frame or bbox is NaN
            behaviors[hid] = "out_of_frame"
            emotions[hid] = "Neutral" # Or "Unknown"

    # Check for social interaction between hippo 1 and hippo 2 if both are active
    # This assumes hippo IDs 1 and 2 are defined in config.HIPPO_PROFILES_CNN
    hippo1_id = 1 
    hippo2_id = 2 
    # Ensure these IDs are actually present in the current state before trying to access
    if hippo1_id in active_hippos_data_for_interaction and hippo2_id in active_hippos_data_for_interaction:
        h1_bbox = active_hippos_data_for_interaction[hippo1_id]
        h2_bbox = active_hippos_data_for_interaction[hippo2_id]

        c1x, c1y = h1_bbox[0] + h1_bbox[2] / 2.0, h1_bbox[1] + h1_bbox[3] / 2.0
        c2x, c2y = h2_bbox[0] + h2_bbox[2] / 2.0, h2_bbox[1] + h2_bbox[3] / 2.0
        dist_between_centers = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
        
        avg_dim_h1 = (h1_bbox[2] + h1_bbox[3]) / 2.0
        avg_dim_h2 = (h2_bbox[2] + h2_bbox[3]) / 2.0
        
        # If distance is less than (e.g., 0.75 times) sum of their average dimensions, consider them interacting
        interaction_distance_threshold = (avg_dim_h1 + avg_dim_h2) * getattr(config, 'INTERACTION_PROXIMITY_FACTOR_CNN', 0.75)

        if dist_between_centers < interaction_distance_threshold:
            # Simple interaction logic: if both relatively slow, calm social. If one or both active, playful social.
            if _velocities_normalized[hippo1_id] < 0.15 and _velocities_normalized[hippo2_id] < 0.15:
                behaviors[hippo1_id] = behaviors[hippo2_id] = "social_interaction_positive" # Or "social_resting"
                emotions[hippo1_id] = emotions[hippo2_id] = "Calm"
            elif (_velocities_normalized[hippo1_id] > 0.3 or _velocities_normalized[hippo2_id] > 0.3):
                behaviors[hippo1_id] = behaviors[hippo2_id] = "social_interaction_positive" # Or "social_active"
                emotions[hippo1_id] = emotions[hippo2_id] = "Playful"
            # Could add "social_interaction_negative" based on very high velocities + proximity, but that's complex.
    
    return behaviors, emotions


if __name__ == '__main__':
    # This block is for basic module testing, not full application run.
    logger.info("Testing cnn_hippo_detector.py module (standalone mode)...")
    
    # 1. Initialize kernels (essential for motion detector if used directly)
    _initialize_kernels_cnn()
    logger.info("Morphological kernels initialized (or confirmed).")

    # 2. Attempt to load a pre-trained CNN model (if one exists)
    if not load_trained_cnn_model():
        logger.info("No pre-trained CNN model found or failed to load. Functions requiring the model might not work fully.")
    else:
        logger.info(f"CNN Model '{config.CNN_MODEL_SAVE_PATH}' loaded successfully for testing.")

    # 3. Example: Test annotation data loading (optional here, done by run_cnn_bbox_annotation_mode)
    # _load_cnn_bbox_annotations()
    # logger.info(f"Attempted to load CNN annotations: {len(_cnn_annot_annotations_data)} entries found.")

    # 4. Suggest how to run key functionalities for testing from here (if desired)
    #    For example, one might want to manually call run_cnn_bbox_annotation_mode()
    #    or train_custom_cnn_detector_model() if sufficient data is present.
    #    However, these are typically run via the main.py menu.
    
    # Example: If you wanted to quickly test the annotation mode:
    # Make sure config.CLIPS_DIR has some .mp4 files.
    # And config.HIPPO_PROFILES_CNN is set up.
    # run_cnn_bbox_annotation_mode() 
    
    # Example: If you wanted to quickly test training (requires annotations and patches):
    # train_custom_cnn_detector_model()

    logger.info("To run the full application workflow with menu options, use: python -m src.main")
    logger.info("Standalone test finished for cnn_hippo_detector.py.")