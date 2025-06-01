# src/annotation_tool.py
import cv2
import os
import logging
import json
import numpy as np
import re 

from src import config # Now directly using your project's config

BBOX_FEATURE_LENGTH = getattr(config, 'BBOX_FEATURE_LENGTH', 5)

logging.basicConfig(level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnotationTool:
    def __init__(self, features_dir, clips_dir, output_annotation_file):
        self.features_dir = features_dir
        self.clips_dir = clips_dir
        self.output_annotation_file = output_annotation_file
        self.annotations = self._load_annotations()

        self.behavior_classes = getattr(config, 'BEHAVIOR_CLASSES', [])
        if not self.behavior_classes or not isinstance(self.behavior_classes, list):
            logger.critical("config.BEHAVIOR_CLASSES error. Check definition in config.py.")
            raise ValueError("Behavior classes not configured correctly.")
        
        self.behavior_key_map_console = {str(i + 1): behavior for i, behavior in enumerate(self.behavior_classes)}
        self.behavior_cv_key_map = {ord(str(i + 1)): behavior for i, behavior in enumerate(self.behavior_classes) if i < 9} # Supports 1-9

        self.emotion_classes = getattr(config, 'EMOTION_CLASSES_FOR_ANNOTATION', [])
        self.annotate_emotions = bool(self.emotion_classes and isinstance(self.emotion_classes, list))
        
        self.emotion_key_map_console = {} 
        self.emotion_cv_key_map = {}      
        
        if self.annotate_emotions:
            emotion_keys_str = ['z', 'x', 'c', 'v', 'b', 'm'] # Max 6 emotion keys
            for i, emotion_class in enumerate(self.emotion_classes):
                if i < len(emotion_keys_str):
                    key_char = emotion_keys_str[i]
                    self.emotion_key_map_console[key_char] = emotion_class
                    self.emotion_cv_key_map[ord(key_char)] = emotion_class
                else:
                    logger.warning(f"More emotion classes ({len(self.emotion_classes)}) than defined keys ({len(emotion_keys_str)}). "
                                   f"Emotion class '{emotion_class}' will not have a direct key binding.")
                    break
            logger.info(f"Emotion annotation ENABLED. Direct CV keys for emotions: {list(self.emotion_key_map_console.keys())}")
        else:
            logger.info("Emotion annotation DISABLED.")

        self.general_command_cv_key_map = {
            ord('n'): 'skip_to_next_file',             
            ord('q'): 'quit_session_fully',              
            ord('p'): 'toggle_pause_play', 
        }
        
        self.hippo_profiles = getattr(config, 'HIPPO_PROFILES_CNN', {})
        if not self.hippo_profiles:
             logger.warning("config.HIPPO_PROFILES_CNN is empty or not defined. Hippo selection might not work.")
        self.hippo_selection_cv_key_map = {} # Maps ord(hippo_profile_key_char) -> hippo_profile_key_int
        for hippo_id_key_from_config, profile_data_init in self.hippo_profiles.items():
            if isinstance(hippo_id_key_from_config, int) and 0 < hippo_id_key_from_config < 10: # User presses '1' through '9'
                self.hippo_selection_cv_key_map[ord(str(hippo_id_key_from_config))] = hippo_id_key_from_config
            if "feature_vector_index" not in profile_data_init:
                logger.error(f"CRITICAL: 'feature_vector_index' is MISSING for hippo ID {hippo_id_key_from_config} in config.HIPPO_PROFILES_CNN. Annotation will likely fail for this hippo.")

        self.display_initial_instructions()

    def _load_annotations(self):
        annotations_list = []
        output_dir = os.path.dirname(self.output_annotation_file)
        if output_dir and not os.path.exists(output_dir):
            try: os.makedirs(output_dir, exist_ok=True)
            except OSError as e: logger.error(f"Could not create dir {output_dir} for annotations: {e}.")
        
        if os.path.exists(self.output_annotation_file):
            try:
                with open(self.output_annotation_file, 'r') as f: content = f.read()
                if content.strip(): annotations_list = json.loads(content)
                logger.info(f"Loaded {len(annotations_list)} annotations from {self.output_annotation_file}.")
            except json.JSONDecodeError: logger.warning(f"Annotation file {self.output_annotation_file} malformed. Starting fresh.")
            except Exception as e: logger.error(f"Error loading annotations from {self.output_annotation_file}: {e}. Starting fresh.")
        else: logger.info(f"Annotation file {self.output_annotation_file} not found. Starting fresh.")
        return annotations_list

    def _save_annotations(self):
        try:
            output_dir = os.path.dirname(self.output_annotation_file)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(self.output_annotation_file):
                backup_file = self.output_annotation_file + ".bak"
                if os.path.exists(backup_file): os.remove(backup_file)
                try: os.rename(self.output_annotation_file, backup_file)
                except OSError: 
                    from shutil import copyfile
                    copyfile(self.output_annotation_file, backup_file)
                    os.remove(self.output_annotation_file)
            with open(self.output_annotation_file, 'w') as f: json.dump(self.annotations, f, indent=4)
            logger.info(f"Annotations saved ({len(self.annotations)}) to {self.output_annotation_file}")
        except Exception as e: logger.error(f"Error saving annotations: {e}")

    def _get_annotation_id(self, feature_filename_base, original_end_frame, hippo_profile_key, original_start_frame):
        # hippo_profile_key is the key from config.HIPPO_PROFILES_CNN (e.g., 1 or 2)
        return f"BEH_{feature_filename_base}::OEF{original_end_frame}::HK{hippo_profile_key}::OSF{original_start_frame}"

    def _display_frame_with_info(self, frame, window_name, text_lines):
        if frame is None:
            logger.error("_display_frame_with_info: received None frame. Cannot display.")
            h_blank = getattr(config, 'DEFAULT_VIDEO_HEIGHT', 480); w_blank = getattr(config, 'DEFAULT_VIDEO_WIDTH', 640)
            blank_frame = np.zeros((h_blank, w_blank, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "ERROR: Frame is None", (w_blank // 4, h_blank // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            try: cv2.imshow(window_name, blank_frame)
            except cv2.error as e: logger.error(f"OpenCV error showing blank frame: {e}")
            return

        display_frame = frame.copy(); y_offset = 25; line_height_px = 20
        for i, line in enumerate(text_lines):
            current_y = y_offset + i * line_height_px
            (text_width, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rect_x1 = 8; rect_y1 = current_y - text_h - 3; rect_x2 = 12 + text_width; rect_y2 = current_y + 4
            text_x = 10; text_y = current_y
            rect_x1=max(0,rect_x1); rect_y1=max(0,rect_y1); rect_x2=min(display_frame.shape[1]-1,rect_x2); rect_y2=min(display_frame.shape[0]-1,rect_y2)
            text_x=max(0,text_x); text_y=max(0,text_y); text_y=min(display_frame.shape[0]-5,text_y)
            if rect_x1 < rect_x2 and rect_y1 < rect_y2 : cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0,0,0), -1)
            cv2.putText(display_frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv2.LINE_AA)
        try: cv2.imshow(window_name, display_frame)
        except cv2.error as e: logger.error(f"OpenCV error in imshow: {e}. Frame shape: {display_frame.shape if display_frame is not None else 'None'}")

    def _get_clip_original_start_frame(self, clip_filename):
        match = re.search(r"_segment_(\d+)_(\d+)", clip_filename)
        if match:
            try: return int(match.group(1))
            except ValueError: logger.warning(f"Could not parse start frame (int) from: {clip_filename}")
        logger.warning(f"Filename format for original start frame not recognized: {clip_filename}. Expected '_segment_START_END'.")
        return None

    def annotate_segment_features(self):
        logger.info("Starting Interactive Annotation Tool for Hippos...")
        if not self.hippo_profiles:
            logger.error("config.HIPPO_PROFILES_CNN is not defined or empty. Cannot select hippos for annotation."); return
        
        if not os.path.isdir(self.features_dir): logger.error(f"Features dir not found: {self.features_dir}. Aborting."); return
        feature_files_all = sorted([f for f in os.listdir(self.features_dir) if f.endswith("_features.json")])
        if not feature_files_all: logger.warning(f"No feature files in {self.features_dir}."); return
        annotated_sequence_ids = {ann.get('annotation_id') for ann in self.annotations}
        logger.info(f"Loaded {len(self.annotations)} existing annotations ({len(annotated_sequence_ids)} unique IDs).")
        window_name = 'Interactive Hippo Annotator'; cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        current_file_list_idx = 0; quit_all_session_flag = False
        paused_for_hippo_selection = False; latest_frame_data = None; selected_hippo_id_to_annotate = None
        
        while current_file_list_idx < len(feature_files_all) and not quit_all_session_flag:
            feature_filename = feature_files_all[current_file_list_idx]
            feature_filepath = os.path.join(self.features_dir, feature_filename)
            feature_filename_base_for_id = os.path.splitext(feature_filename)[0] 
            clip_filename_base_for_match = feature_filename.replace("_features.json", "")
            clip_filepath = None; video_extensions = getattr(config, 'VIDEO_EXTENSIONS', ('.mp4', '.avi', '.mov'))
            for ext in video_extensions:
                potential_clip_path = os.path.join(self.clips_dir, clip_filename_base_for_match + ext)
                if os.path.exists(potential_clip_path): clip_filepath = potential_clip_path; break
            if not clip_filepath: logger.warning(f"Clip for {feature_filename} not found. Skipping."); current_file_list_idx += 1; continue
            
            clip_original_start_frame = self._get_clip_original_start_frame(os.path.basename(clip_filepath))
            if clip_original_start_frame is None: logger.error(f"Cannot get original start for {os.path.basename(clip_filepath)}. Skipping."); current_file_list_idx += 1; continue
            
            try:
                with open(feature_filepath, 'r') as f: segment_frame_features_data = json.load(f)
            except Exception as e: logger.error(f"Error loading {feature_filepath}: {e}. Skipping."); current_file_list_idx += 1; continue
            
            if not segment_frame_features_data or not isinstance(segment_frame_features_data, list) or \
               len(segment_frame_features_data) == 0 or not isinstance(segment_frame_features_data[0], dict) or \
               'feature_vectors' not in segment_frame_features_data[0] or \
               'frame_idx' not in segment_frame_features_data[0]:
                logger.warning(f"Features {feature_filepath} empty/invalid or first entry misses 'feature_vectors' or 'frame_idx'. Skipping."); 
                current_file_list_idx += 1; continue # No cap.release() needed yet
            
            cap = cv2.VideoCapture(clip_filepath)
            if not cap.isOpened(): logger.error(f"Cannot open clip: {clip_filepath}. Skipping."); current_file_list_idx += 1; continue
            
            clip_fps = cap.get(cv2.CAP_PROP_FPS); clip_frame_delay = int(1000 / clip_fps) if clip_fps > 0 else 30
            total_frames_in_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); current_clip_frame_num = -1
            logger.info(f"\n--- Playing File {current_file_list_idx + 1}/{len(feature_files_all)}: {os.path.basename(clip_filepath)} ({total_frames_in_clip} frames) ---")
            
            skip_to_next_file_flag = False
            ret_init, frame_init = cap.read()
            if not ret_init: logger.warning(f"Could not read initial frame from {clip_filepath}. Skipping."); cap.release(); current_file_list_idx += 1; continue
            latest_frame_data = frame_init.copy(); current_clip_frame_num = 0

            # --- CLIP PLAYBACK & ANNOTATION LOOP ---
            while True: 
                if not paused_for_hippo_selection and selected_hippo_id_to_annotate is None: 
                    if current_clip_frame_num < total_frames_in_clip -1 :
                        ret_read, frame_read_next = cap.read()
                        if not ret_read: break 
                        latest_frame_data = frame_read_next.copy()
                        current_clip_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) -1 
                        if current_clip_frame_num < 0: current_clip_frame_num = 0 
                    else: break 
                
                frame_to_display = latest_frame_data 
                if frame_to_display is None: break

                info_text_lines = [f"Clip: {os.path.basename(clip_filepath)} | Frame: {current_clip_frame_num}/{total_frames_in_clip-1} (Orig.Start: {clip_original_start_frame})"]
                if paused_for_hippo_selection:
                    if selected_hippo_id_to_annotate is not None: 
                        hippo_profile_data = self.hippo_profiles.get(selected_hippo_id_to_annotate, {})
                        hippo_name_disp = hippo_profile_data.get("name_short", hippo_profile_data.get("name", f"Hippo {selected_hippo_id_to_annotate}"))
                        info_text_lines.append(f"ANNOTATING {hippo_name_disp.upper()}:")
                        bhv_keys_str = ",".join([chr(k) for k in sorted(self.behavior_cv_key_map.keys())])
                        info_text_lines.append(f"  BEHAVIOR Keys: [{bhv_keys_str}]")
                        if self.annotate_emotions:
                            emo_keys_str = ",".join([chr(k) for k in sorted(self.emotion_cv_key_map.keys())])
                            info_text_lines.append(f"  EMOTION Keys: [{emo_keys_str}] (or [S]kipEmo)")
                        info_text_lines.append(f"  Cmds: [N]extFile [Q]uit | (Finish this hippo to return to HippoSelect)")
                    else: 
                        info_text_lines.append("PAUSED - SELECT HIPPO TO ANNOTATE:")
                        hippo_select_prompt = []
                        for key_code_select, hippo_id_val_select in self.hippo_selection_cv_key_map.items():
                            name_select = self.hippo_profiles.get(hippo_id_val_select, {}).get("name_short", f"H{hippo_id_val_select}")
                            hippo_select_prompt.append(f"[{chr(key_code_select)}]{name_select}")
                        if hippo_select_prompt: info_text_lines.append("  " + " ".join(hippo_select_prompt))
                        else: info_text_lines.append("  No hippos (1-9) found in config.HIPPO_PROFILES_CNN.")
                        info_text_lines.append("  Cmds: [P]/[SPACE]Resume [N]extFile [Q]uit [F]NextFrame")
                else: 
                    info_text_lines.append("PLAYING: [SPACE]Pause&SelectHippo [N]extFile [Q]uit")
                self._display_frame_with_info(frame_to_display, window_name, info_text_lines)
                
                wait_duration = 0 if paused_for_hippo_selection else clip_frame_delay
                key_code = cv2.waitKey(wait_duration) & 0xFF

                if key_code != 255: 
                    logger.debug(f"Key: '{chr(key_code) if key_code < 128 else key_code}' ({key_code}) | PausedSel: {paused_for_hippo_selection} | AnnotatingID: {selected_hippo_id_to_annotate}")

                    if key_code in self.general_command_cv_key_map:
                        command = self.general_command_cv_key_map[key_code]
                        if command == 'skip_to_next_file': skip_to_next_file_flag = True
                        elif command == 'quit_session_fully': quit_all_session_flag = True
                        elif command == 'toggle_pause_play':
                            if paused_for_hippo_selection:
                                paused_for_hippo_selection = False; selected_hippo_id_to_annotate = None
                                logger.info("Resumed playback by 'P'.")
                        if skip_to_next_file_flag or quit_all_session_flag:
                             paused_for_hippo_selection = False; selected_hippo_id_to_annotate = None; break 
                        continue 

                    if paused_for_hippo_selection:
                        if selected_hippo_id_to_annotate is not None: # --- ANNOTATING SELECTED HIPPO ---
                            chosen_behavior = None; final_emotion = "Not_Annotated"
                            
                            if key_code in self.behavior_cv_key_map:
                                chosen_behavior = self.behavior_cv_key_map[key_code]
                                current_hippo_profile_data = self.hippo_profiles.get(selected_hippo_id_to_annotate, {})
                                current_hippo_name_disp = current_hippo_profile_data.get("name_short", hippo_profile_data.get("name", f"H{selected_hippo_id_to_annotate}"))
                                logger.info(f"For {current_hippo_name_disp}, Behavior: '{chosen_behavior}'")

                                if self.annotate_emotions:
                                    temp_emo_prompt_lines = list(info_text_lines) 
                                    emo_keys_str_disp = ",".join([chr(k) for k in sorted(self.emotion_cv_key_map.keys())])
                                    temp_emo_prompt_lines[1] = f"ANNOTATING {current_hippo_name_disp.upper()} (Beh: {chosen_behavior})" 
                                    temp_emo_prompt_lines[2] = f"  EMOTION Keys: [{emo_keys_str_disp}] (or [S]kipEmo)"
                                    temp_emo_prompt_lines[3] = "  Cmds for Emo: [N]extFile [Q]uit" 
                                    self._display_frame_with_info(frame_to_display, window_name, temp_emo_prompt_lines)
                                    
                                    emo_key_code = cv2.waitKey(0) & 0xFF
                                    logger.debug(f"Emotion key: '{chr(emo_key_code) if emo_key_code < 128 else emo_key_code}' ({emo_key_code})")
                                    if emo_key_code in self.emotion_cv_key_map:
                                        final_emotion = self.emotion_cv_key_map[emo_key_code]; logger.info(f"Emotion: '{final_emotion}'")
                                    elif emo_key_code == ord('s'): logger.info("Emotion skipped.")
                                    elif emo_key_code in self.general_command_cv_key_map: 
                                        emo_command = self.general_command_cv_key_map[emo_key_code]
                                        if emo_command == 'skip_to_next_file': skip_to_next_file_flag = True
                                        elif emo_command == 'quit_session_fully': quit_all_session_flag = True
                                    else: logger.warning(f"Invalid key '{chr(emo_key_code)}' for emotion. Using 'Not_Annotated'.")
                                
                                if skip_to_next_file_flag or quit_all_session_flag:
                                    selected_hippo_id_to_annotate = None; paused_for_hippo_selection = False; break

                                target_orig_end_fr = clip_original_start_frame + current_clip_frame_num
                                target_orig_start_fr = target_orig_end_fr - config.SEQUENCE_LENGTH + 1
                                
                                selected_hippo_profile = self.hippo_profiles.get(selected_hippo_id_to_annotate) # Should exist
                                h_idx_for_features = selected_hippo_profile.get("feature_vector_index") # Get from config
                                hippo_name_for_ann_entry = selected_hippo_profile.get("name", f"Hippo_{selected_hippo_id_to_annotate}")

                                if h_idx_for_features is None:
                                    logger.error(f"'feature_vector_index' MISSING for hippo ID {selected_hippo_id_to_annotate} in config.HIPPO_PROFILES_CNN. Cannot save annotation.")
                                else:
                                    logger.debug(f"Using feature_vector_index: {h_idx_for_features} for hippo ID {selected_hippo_id_to_annotate}")
                                    seq_features_for_selected_hippo, actual_orig_frames_in_seq = [], []
                                    for frame_data_entry in segment_frame_features_data:
                                        orig_frame_idx_json = frame_data_entry.get('frame_idx', -1)
                                        if target_orig_start_fr <= orig_frame_idx_json <= target_orig_end_fr:
                                            frame_feature_vectors = frame_data_entry.get('feature_vectors', [])
                                            if h_idx_for_features < len(frame_feature_vectors):
                                                fv = frame_feature_vectors[h_idx_for_features]
                                                if isinstance(fv, list) and len(fv) == BBOX_FEATURE_LENGTH:
                                                    seq_features_for_selected_hippo.append(fv)
                                                    actual_orig_frames_in_seq.append(orig_frame_idx_json)
                                                else: 
                                                    seq_features_for_selected_hippo.append([np.nan]*BBOX_FEATURE_LENGTH); actual_orig_frames_in_seq.append(orig_frame_idx_json)
                                                    logger.debug(f"  Malformed FV for H_FV_IDX {h_idx_for_features} at orig_frame {orig_frame_idx_json}")
                                            else: 
                                                seq_features_for_selected_hippo.append([np.nan]*BBOX_FEATURE_LENGTH); actual_orig_frames_in_seq.append(orig_frame_idx_json)
                                                logger.debug(f"  H_FV_IDX {h_idx_for_features} out of bounds for feature_vectors (len {len(frame_feature_vectors)}) at orig_frame {orig_frame_idx_json}")
                                    
                                    logger.debug(f"  For hippo ID {selected_hippo_id_to_annotate} (FV_idx {h_idx_for_features}), collected {len(seq_features_for_selected_hippo)} feature sets. "
                                                 f"Target original frames: {target_orig_start_fr}-{target_orig_end_fr}. "
                                                 f"Actual original frames found: {actual_orig_frames_in_seq[:3]}...{actual_orig_frames_in_seq[-3:] if len(actual_orig_frames_in_seq)>5 else actual_orig_frames_in_seq}")

                                    if len(seq_features_for_selected_hippo) >= config.SEQUENCE_LENGTH:
                                        # Sort by original frame number and select the correct segment
                                        collected_pairs = sorted(zip(actual_orig_frames_in_seq, seq_features_for_selected_hippo), key=lambda x:x[0])
                                        valid_end_idx_in_collected = -1
                                        for i_pair in range(len(collected_pairs)-1, -1, -1):
                                            if collected_pairs[i_pair][0] <= target_orig_end_fr: # Find first frame at or before target end
                                                valid_end_idx_in_collected = i_pair; break
                                        
                                        if valid_end_idx_in_collected != -1 and (valid_end_idx_in_collected - config.SEQUENCE_LENGTH + 1 >=0):
                                            start_slice_idx = valid_end_idx_in_collected - config.SEQUENCE_LENGTH + 1
                                            final_sel_pairs = collected_pairs[start_slice_idx : valid_end_idx_in_collected + 1]

                                            if len(final_sel_pairs) == config.SEQUENCE_LENGTH:
                                                final_seq_frames = [p[0] for p in final_sel_pairs]; final_seq_features = [p[1] for p in final_sel_pairs]
                                                seq_start_orig_id = final_seq_frames[0]; seq_end_orig_id = final_seq_frames[-1]
                                                seq_end_ts_id = -1.0; 
                                                for fd_entry in segment_frame_features_data: 
                                                    if fd_entry['frame_idx'] == seq_end_orig_id: seq_end_ts_id = fd_entry.get('timestamp_sec', -1.0); break
                                                
                                                # Use selected_hippo_id_to_annotate (the key from config like 1 or 2) for the ID
                                                ann_id = self._get_annotation_id(feature_filename_base_for_id, seq_end_orig_id, selected_hippo_id_to_annotate, seq_start_orig_id)
                                                
                                                if ann_id not in annotated_sequence_ids and not np.all(np.isnan(np.array(final_seq_features))):
                                                    flat_feats_json = np.array(final_seq_features).flatten().tolist()
                                                    orig_vid_path_ref = segment_frame_features_data[0].get('video_path', 'unknown_orig.mp4')
                                                    
                                                    annotation_entry = {
                                                        'annotation_id': ann_id, 'feature_file_source': feature_filename,
                                                        'clip_path_annotated_from': clip_filepath, 'original_video_path': orig_vid_path_ref, 
                                                        'hippo_profile_key': selected_hippo_id_to_annotate, 
                                                        'hippo_name': hippo_name_for_ann_entry,
                                                        'sequence_start_original_frame': seq_start_orig_id,
                                                        'sequence_end_original_frame': seq_end_orig_id, 
                                                        'timestamp_at_label_end': seq_end_ts_id, 'features': flat_feats_json, 
                                                        'label': chosen_behavior, 'emotion': final_emotion }
                                                    self.annotations.append(annotation_entry); annotated_sequence_ids.add(ann_id)
                                                    self._save_annotations()
                                                    logger.info(f"SAVED: {ann_id} | For: {hippo_name_for_ann_entry} | Beh='{chosen_behavior}', Emo='{final_emotion}'")
                                                elif ann_id in annotated_sequence_ids: logger.info(f"Seq {ann_id} for {hippo_name_for_ann_entry} already annotated.")
                                                else: logger.info(f"Seq for {hippo_name_for_ann_entry} (orig end {seq_end_orig_id}) is all NaNs.")
                                            else: logger.info(f"Could not form exact sequence length for {hippo_name_for_ann_entry} after slicing. Found {len(final_sel_pairs)}.")
                                        else: logger.info(f"Not enough valid collected features to form sequence for {hippo_name_for_ann_entry} ending at or before {target_orig_end_fr}.")
                                    else: logger.info(f"Not enough features initially collected ({len(seq_features_for_selected_hippo)}) for {hippo_name_for_ann_entry} "
                                                      f"for target range {target_orig_start_fr}-{target_orig_end_fr}.")
                                
                                selected_hippo_id_to_annotate = None # Done with this hippo, return to hippo selection (still paused)
                                logger.info("Annotation attempt for selected hippo finished. Returned to Hippo Selection mode (still paused).")
                            # else: An invalid key (not behavior) was pressed while expecting behavior for this hippo.
                            # The loop will redraw the prompt. No specific action needed here.
                            # General commands N, Q, P are handled above this 'if paused_for_hippo_selection:' block.

                        elif selected_hippo_id_to_annotate is None: # --- PAUSED, WAITING FOR HIPPO SELECTION ---
                            if key_code in self.hippo_selection_cv_key_map:
                                selected_hippo_id_to_annotate = self.hippo_selection_cv_key_map[key_code]
                                hippo_name_sel = self.hippo_profiles.get(selected_hippo_id_to_annotate, {}).get("name_short", self.hippo_profiles.get(selected_hippo_id_to_annotate, {}).get("name", f"H{selected_hippo_id_to_annotate}"))
                                logger.info(f"Selected {hippo_name_sel} for annotation. Now input behavior.")
                            elif key_code == ord('f'): 
                                if current_clip_frame_num < total_frames_in_clip -1:
                                    ret_f, frame_f_next = cap.read()
                                    if ret_f: latest_frame_data = frame_f_next.copy(); current_clip_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) -1
                                    else: paused_for_hippo_selection = False; skip_to_next_file_flag = True 
                                else: logger.info("At last frame of clip.")
                            elif key_code == ord(' '): 
                                paused_for_hippo_selection = False; selected_hippo_id_to_annotate = None; 
                                logger.info("Playback resumed by Spacebar from hippo selection.")
                            else: logger.debug(f"Unmapped key '{chr(key_code)}' while in Hippo Selection mode.")
                    
                    else: # --- PLAYING ---
                        if key_code == ord(' '): 
                            paused_for_hippo_selection = True
                            selected_hippo_id_to_annotate = None 
                            logger.info(f"PAUSED for Hippo Selection at clip frame: {current_clip_frame_num}")
                        # N and Q are handled by the general command check block at the start of key handling
                        # else: logger.debug(f"Unmapped key '{chr(key_code)}' during playback.") # Redundant if N,Q handled
                
                # Loop termination conditions
                if current_clip_frame_num >= total_frames_in_clip -1 and not paused_for_hippo_selection: 
                    logger.debug(f"Reached end of clip {os.path.basename(clip_filepath)} naturally during playback.")
                    break 
                if skip_to_next_file_flag or quit_all_session_flag: break
            
            cap.release(); paused_for_hippo_selection = False; selected_hippo_id_to_annotate = None
            if quit_all_session_flag: break 
            current_file_list_idx += 1
            
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow(window_name)
        except cv2.error: pass
        logger.info("Annotation process finished or user quit."); self._save_annotations()

    def display_initial_instructions(self):
        logger.info("--- Interactive Multi-Hippo Annotation Tool ---")
        logger.info("Video will play. Press [SPACE] to PAUSE and enter HIPPO SELECTION mode.")
        logger.info("--- When PAUSED for HIPPO SELECTION: ---")
        hippo_select_prompts = []
        # Use self.hippo_selection_cv_key_map which maps ord(char_key) to hippo_id_from_config
        for key_code_ord, hippo_id_val_prof in self.hippo_selection_cv_key_map.items():
            profile_data_instr = self.hippo_profiles.get(hippo_id_val_prof, {}) # Use hippo_id_val_prof to get profile
            name_instr = profile_data_instr.get("name_short", profile_data_instr.get("name", f"H{hippo_id_val_prof}"))
            hippo_select_prompts.append(f"[{chr(key_code_ord)}]{name_instr}") 
        if hippo_select_prompts:
            logger.info(f"  Select Hippo using its number key: {' '.join(hippo_select_prompts)}")
        else:
            logger.warning("No hippo selection keys configured (check config.HIPPO_PROFILES_CNN keys are 1-9 int).")
        logger.info("  Other commands: [P]/[SPACE]ResumePlay [F]NextFrameWhilePaused")
        
        logger.info("--- Once a HIPPO IS SELECTED (still paused): ---")
        bhv_keys_str_instr = ",".join([chr(k) for k in sorted(self.behavior_cv_key_map.keys())])
        logger.info(f"  BEHAVIOR: Press a behavior key [{bhv_keys_str_instr}] for the selected hippo.")
        if self.annotate_emotions:
            emo_keys_str_instr = ",".join([chr(k) for k in sorted(self.emotion_cv_key_map.keys())])
            logger.info(f"  EMOTION (after behavior): Press an emotion key [{emo_keys_str_instr}] or [S] to skip emotion.")
        logger.info("  After annotating a hippo, you return to HIPPO SELECTION at the same frame.")
        
        logger.info("--- General Commands (available in most states): ---")
        logger.info("    [N] : Skip to Next Video File.")
        logger.info("    [Q] : Quit Entire Session.")
        
        logger.info("--- Behavior Classes (for reference when using number keys): ---")
        for key_num_str_instr, val_beh_instr in self.behavior_key_map_console.items(): 
            logger.info(f"  Press '{key_num_str_instr}' for '{val_beh_instr}'")
        if self.annotate_emotions:
            logger.info("--- Emotion Classes (for reference when using letter keys): ---")
            for key_char_str_instr, val_emo_instr in self.emotion_key_map_console.items(): 
                logger.info(f"  Press '{key_char_str_instr.upper()}' for '{val_emo_instr}'")


# --- __main__ block (ensure config.HIPPO_PROFILES_CNN has "feature_vector_index") ---
if __name__ == '__main__':
    # This __main__ block should use the config object that's actually in scope.
    # The MinimalConfig fallback at the top handles the case where `from src import config` fails.
    
    # Ensure HIPPO_PROFILES_CNN in the effective config has "feature_vector_index"
    # This is a patch for the MinimalConfig for testing this specific feature.
    # In your actual project, config.py should define this.
    if not hasattr(config, 'HIPPO_PROFILES_CNN') or not config.HIPPO_PROFILES_CNN:
        logger.error("CRITICAL FOR TEST: config.HIPPO_PROFILES_CNN is not defined or empty. Test may fail or be meaningless.")
    else:
        missing_fv_idx = False
        for h_id, h_data in config.HIPPO_PROFILES_CNN.items():
            if "feature_vector_index" not in h_data:
                logger.warning(f"TESTING WARNING: Hippo ID {h_id} in config.HIPPO_PROFILES_CNN is missing 'feature_vector_index'. "
                               "Attempting to assign based on order if possible, or default to 0 for first, 1 for second etc.")
                missing_fv_idx = True
        if missing_fv_idx and isinstance(config, MinimalConfig): # Only patch if it's our internal fallback
             # Simple assignment for MinimalConfig if feature_vector_index is missing
            temp_idx = 0
            for h_id_patch in sorted(config.HIPPO_PROFILES_CNN.keys()): # Sort to make assignment consistent
                if "feature_vector_index" not in config.HIPPO_PROFILES_CNN[h_id_patch]:
                     config.HIPPO_PROFILES_CNN[h_id_patch]["feature_vector_index"] = temp_idx
                     logger.info(f"Patched MinimalConfig: Hippo ID {h_id_patch} assigned feature_vector_index {temp_idx}")
                     temp_idx +=1


    # Rest of the __main__ block for dummy data creation and running the tool...
    test_base_dir = getattr(config, 'PROCESSED_DATA_DIR', os.path.join(os.path.dirname(__file__), "..", "data_test_annot_tool"))
    dummy_features_dir = getattr(config, 'FEATURES_DIR', os.path.join(test_base_dir, "dummy_features_multi"))
    dummy_clips_dir = getattr(config, 'CLIPS_DIR', os.path.join(test_base_dir, "dummy_clips_multi"))
    dummy_annotations_output_dir = os.path.join(test_base_dir, "dummy_annotations_output_multi")
    os.makedirs(dummy_features_dir, exist_ok=True); os.makedirs(dummy_clips_dir, exist_ok=True); os.makedirs(dummy_annotations_output_dir, exist_ok=True)
    output_annotations_json_path_test = os.path.join(dummy_annotations_output_dir, "TEST_interactive_multihippo_annotations.json")
    example_clip_name = "dummy_vid_multi_h_test"
    original_start_f = 1500 
    sequence_len_for_test = getattr(config, 'SEQUENCE_LENGTH', 10)
    num_frames_in_dummy_clip = sequence_len_for_test * 8
    example_feature_file_name = f"{example_clip_name}_segment_{original_start_f}_{original_start_f + num_frames_in_dummy_clip -1}_features.json"
    dummy_features_path = os.path.join(dummy_features_dir, example_feature_file_name)
    if not os.path.exists(dummy_features_path):
        logger.info(f"Creating dummy features for test: {dummy_features_path}")
        num_hippo_profiles_to_simulate = 0
        if hasattr(config, 'HIPPO_PROFILES_CNN') and isinstance(config.HIPPO_PROFILES_CNN, dict):
            # Count profiles that have a feature_vector_index defined, as this is what feature_vectors will correspond to
            num_hippo_profiles_to_simulate = len([pid for pid, pdata in config.HIPPO_PROFILES_CNN.items() if "feature_vector_index" in pdata])
        if num_hippo_profiles_to_simulate == 0: num_hippo_profiles_to_simulate = 1 
        logger.info(f"Simulating {num_hippo_profiles_to_simulate} hippos in dummy feature file based on config.")
        
        dummy_segment_feature_data = []
        for i in range(num_frames_in_dummy_clip): 
            feature_vectors_for_frame = []
            for h_prof_idx_sim in range(num_hippo_profiles_to_simulate): 
                nan_probability = 0.15 + (h_prof_idx_sim * 0.15) 
                fv = np.random.rand(BBOX_FEATURE_LENGTH).tolist() if np.random.rand() > nan_probability else [np.nan] * BBOX_FEATURE_LENGTH
                feature_vectors_for_frame.append(fv)
            dummy_segment_feature_data.append({'video_path': f'dummy_orig_{example_clip_name}.mp4', 'frame_idx': original_start_f + i, 'timestamp_sec': (original_start_f + i) * (1.0 / 25.0), 'feature_vectors': feature_vectors_for_frame })
        try:
            with open(dummy_features_path, 'w') as f_out: json.dump(dummy_segment_feature_data, f_out, indent=4)
            logger.info(f"Dummy features for original frames {original_start_f}-{original_start_f + num_frames_in_dummy_clip -1} created with {num_hippo_profiles_to_simulate} FV sets per frame.")
        except Exception as e: logger.error(f"Error creating dummy features: {e}")

    dummy_clip_filename = f"{example_clip_name}_segment_{original_start_f}_{original_start_f + num_frames_in_dummy_clip -1}.mp4"
    dummy_clip_path = os.path.join(dummy_clips_dir, dummy_clip_filename)
    if not os.path.exists(dummy_clip_path) and os.path.exists(dummy_features_path):
        logger.info(f"Creating dummy clip: {dummy_clip_path} ({num_frames_in_dummy_clip} frames)")
        frame_width, frame_height = 640, 480; dummy_video_fps = 25 
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out_dummy_clip = cv2.VideoWriter(dummy_clip_path, fourcc, dummy_video_fps, (frame_width, frame_height))
            for i_frame_clip in range(num_frames_in_dummy_clip): 
                frame_ph = np.full((frame_height, frame_width, 3), (min(255,i_frame_clip * 2), min(255,i_frame_clip * 1), 30 + i_frame_clip % 100), dtype=np.uint8)
                cv2.putText(frame_ph, f"CLIP Fr {i_frame_clip} (Orig {original_start_f + i_frame_clip})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                out_dummy_clip.write(frame_ph)
            out_dummy_clip.release(); logger.info(f"Dummy clip {dummy_clip_path} created.")
        except Exception as e: logger.error(f"Error creating dummy clip: {e}")

    if os.path.exists(dummy_features_dir) and len(os.listdir(dummy_features_dir)) > 0 and \
       os.path.exists(dummy_clips_dir) and len(os.listdir(dummy_clips_dir)) > 0 :
        logger.info(f"--- Starting AnnotationTool Test ---")
        test_tool = AnnotationTool(features_dir=dummy_features_dir, clips_dir=dummy_clips_dir, output_annotation_file=output_annotations_json_path_test)
        test_tool.annotate_segment_features(); logger.info(f"--- AnnotationTool Test Finished. Check output: {output_annotations_json_path_test} ---")
    else:
        logger.error("Dummy feature files or clip files were not found/created. Cannot run test.")
        if not os.path.exists(dummy_features_dir) or not os.listdir(dummy_features_dir): logger.error(f"Problem with features directory: {dummy_features_dir}")
        if not os.path.exists(dummy_clips_dir) or not os.listdir(dummy_clips_dir): logger.error(f"Problem with clips directory: {dummy_clips_dir}")