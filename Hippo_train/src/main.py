# src/main.py
import os
import logging
import json
import pandas as pd
import numpy as np
import cv2 # Added for VideoCapture in cnn_detection_and_feature_pipeline

from src import config # Ensure this is imported first
from src.video_processor import VideoProcessor
from src.cnn_hippo_detector import (
    CustomCNNDetector,
    run_cnn_bbox_annotation_mode,
    train_custom_cnn_detector_model,
    infer_behavior_and_emotion_cnn, # Rule-based behavior from your script
    load_trained_cnn_model as load_cnn_detector_model # For checking if model exists
)
from src.feature_extractor import FeatureExtractor
from src.behavior_classifier import BehaviorClassifier
from src.gemini_handler import GeminiHandler
from src.annotation_tool import AnnotationTool

logging.basicConfig(level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dotenv_at_root():
    dotenv_path = os.path.join(config.BASE_DIR, ".env")
    if not os.path.exists(dotenv_path):
        logger.warning(f"'.env' file not found at {dotenv_path}. Creating a dummy .env file.")
        logger.warning("Please replace 'YOUR_ACTUAL_API_KEY_HERE' with your actual Gemini API key.")
        try:
            with open(dotenv_path, "w") as f:
                f.write("GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE\n")
            from importlib import reload # Careful with reload in complex apps
            try:
                reload(config)
                logger.info("Config reloaded after .env creation.")
            except Exception as e_reload:
                logger.error(f"Could not reload config after .env creation: {e_reload}")
        except Exception as e:
            logger.error(f"Could not create .env file at {dotenv_path}: {e}")

def generate_clips_pipeline(video_processor, video_files_to_process):
    # ... (This function from previous main.py is fine) ...
    logger.info(f"Starting CLIPS ONLY generation for {len(video_files_to_process)} video files.")
    clips_generated_count = 0
    for video_file_path in video_files_to_process:
        video_basename = os.path.basename(video_file_path)
        logger.info(f"\n--- Processing video for clips: {video_basename} ---")
        active_segments = video_processor.extract_active_segments(video_file_path)
        if not active_segments:
            logger.info(f"No active segments found by VideoProcessor in {video_basename}. Skipping clip extraction.")
            continue
        for i, segment_info in enumerate(active_segments):
            segment_id_base = f"{os.path.splitext(video_basename)[0]}_segment_{segment_info['start_frame']}_{segment_info['end_frame']}"
            clip_output_path = os.path.join(config.CLIPS_DIR, f"{segment_id_base}.mp4")
            if not os.path.exists(clip_output_path):
                if video_processor.extract_clip(segment_info['video_path'],
                                                segment_info['start_frame'],
                                                segment_info['end_frame'],
                                                clip_output_path):
                    logger.info(f"    Extracted clip: {os.path.basename(clip_output_path)}")
                    clips_generated_count +=1
        logger.info(f"Finished clip processing for {video_basename}")
    logger.info(f"Clip generation finished. {clips_generated_count} new clips created in: {config.CLIPS_DIR}")
    logger.info("Next: Annotate BBoxes for CNN training (Option 1B) using these clips.")


def cnn_detection_and_feature_pipeline(cnn_detector, feature_extractor, original_video_paths):
    logger.info(f"Starting CNN detection & feature extraction using clips derived from {len(original_video_paths)} original videos.")
    
    processed_clips_count = 0
    # We need to find clips that correspond to the original_video_paths
    # This assumes clips are named consistently: original_video_name_segment_start_end.mp4
    for original_vid_path in original_video_paths:
        original_vid_basename_no_ext = os.path.splitext(os.path.basename(original_vid_path))[0]
        clips_for_this_original_video = [
            os.path.join(config.CLIPS_DIR, f) 
            for f in os.listdir(config.CLIPS_DIR) 
            if f.startswith(original_vid_basename_no_ext) and f.lower().endswith(config.VIDEO_EXTENSIONS)
        ]

        if not clips_for_this_original_video:
            logger.info(f"No clips found in {config.CLIPS_DIR} corresponding to original video {os.path.basename(original_vid_path)}.")
            continue

        for clip_path in clips_for_this_original_video:
            clip_basename = os.path.basename(clip_path)
            logger.info(f"\n--- CNN Detecting & Feature Extracting on clip: {clip_basename} ---")
            
            segment_id_base_for_output = os.path.splitext(clip_basename)[0]
            detections_data_file = os.path.join(config.DETECTIONS_DATA_DIR, f"{segment_id_base_for_output}_detections.json")
            features_data_file = os.path.join(config.DETECTIONS_DATA_DIR, f"{segment_id_base_for_output}_features.json")

            # Reconstruct segment_info for cnn_detector (if it needs original video context)
            # This parsing relies on the clip naming convention: originalname_segment_startframe_endframe.mp4
            try:
                parts = segment_id_base_for_output.split('_segment_')
                # original_video_name_from_clip = parts[0] # Not needed if we have original_vid_path
                frame_info_parts = parts[1].split('_')
                start_f = int(frame_info_parts[0])
                end_f = int(frame_info_parts[1])
                
                temp_cap_orig = cv2.VideoCapture(original_vid_path) # Get FPS from original video
                original_fps_val = temp_cap_orig.get(cv2.CAP_PROP_FPS)
                temp_cap_orig.release()
                if original_fps_val == 0: original_fps_val = 25.0

                current_segment_info = {
                    'video_path': original_vid_path,
                    'start_frame': start_f,
                    'end_frame': end_f,
                    'fps': original_fps_val
                }
            except Exception as e_parse:
                logger.error(f"Could not parse segment info from clip name {clip_basename} for original video context: {e_parse}. Using clip as its own context.")
                # Fallback: treat clip as a standalone video for detection timestamps
                temp_cap_clip = cv2.VideoCapture(clip_path)
                clip_fps_val = temp_cap_clip.get(cv2.CAP_PROP_FPS)
                clip_total_frames = int(temp_cap_clip.get(cv2.CAP_PROP_FRAME_COUNT))
                temp_cap_clip.release()
                if clip_fps_val == 0: clip_fps_val = 25.0
                current_segment_info = {
                    'video_path': clip_path, # Use clip path itself
                    'start_frame': 0,       # Relative to clip
                    'end_frame': clip_total_frames -1, # Relative to clip
                    'fps': clip_fps_val
                }


            segment_detections_data = None
            if os.path.exists(detections_data_file):
                logger.info(f"    Loading existing CNN detections: {os.path.basename(detections_data_file)}")
                try:
                    with open(detections_data_file, 'r') as f: segment_detections_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"    Error decoding JSON from {detections_data_file}. Will reprocess."); segment_detections_data = None
            
            if not segment_detections_data:
                segment_detections_data_raw = cnn_detector.process_segment_for_detections(current_segment_info)
                if segment_detections_data_raw:
                    segment_detections_data = segment_detections_data_raw
                    with open(detections_data_file, 'w') as f: json.dump(segment_detections_data, f, indent=4)
                    logger.info(f"    Saved CNN detections to {os.path.basename(detections_data_file)}")
                else:
                    logger.warning(f"    No CNN detections for clip {clip_basename}. Skipping feature extraction."); continue
            
            if not segment_detections_data: continue # Should be caught above

            temp_clip_cap = cv2.VideoCapture(clip_path)
            clip_frame_w = int(temp_clip_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            clip_frame_h = int(temp_clip_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            temp_clip_cap.release()

            if os.path.exists(features_data_file):
                logger.info(f"    Features file already exists: {os.path.basename(features_data_file)}. Skipping extraction.")
            else:
                logger.info(f"    Extracting features from CNN detections for {clip_basename}...")
                # Pass actual frame dimensions of the clip being processed for normalization
                segment_features_data_raw = feature_extractor.process_segment_data_for_features(
                    segment_detections_data, frame_width=clip_frame_w, frame_height=clip_frame_h
                )
                if segment_features_data_raw:
                    # ... (serialization logic from your previous main.py is fine) ...
                    serializable_features = []
                    for frame_feature_info in segment_features_data_raw:
                        entry_copy = frame_feature_info.copy()
                        if 'feature_vectors' in entry_copy and entry_copy['feature_vectors'] is not None:
                            entry_copy['feature_vectors'] = [fv.tolist() if isinstance(fv, np.ndarray) else (fv if isinstance(fv, list) else []) for fv in entry_copy['feature_vectors']]
                        serializable_features.append(entry_copy)
                    with open(features_data_file, 'w') as f: json.dump(serializable_features, f, indent=4)
                    logger.info(f"    Saved feature data to {os.path.basename(features_data_file)}")
                else: logger.warning(f"    No features extracted for {clip_basename}.")
            processed_clips_count +=1
    logger.info(f"CNN Detection & Feature Extraction finished. Processed {processed_clips_count} clips.")


def convert_behavior_annotations_to_training_csv(json_behavior_annotation_file, output_csv_file):
    # ... (This function from your previous main.py is fine) ...
    logger.info(f"Converting Behavior JSON annotations from '{json_behavior_annotation_file}' to CSV '{output_csv_file}'")
    if not os.path.exists(json_behavior_annotation_file):
        logger.error(f"Behavior Annotation JSON file not found: {json_behavior_annotation_file}"); return False
    try:
        with open(json_behavior_annotation_file, 'r') as f: annotations_data = json.load(f) 
    except Exception as e: logger.error(f"Error reading Behavior Annotation JSON: {e}"); return False
    if not annotations_data: logger.warning(f"No Behavior annotations found. CSV not created."); return False
    training_data_list_for_df = []
    for ann_entry in annotations_data:
        if not all(k in ann_entry for k in ['video_path', 'sequence_end_original_frame', 
                                            'timestamp_at_label_end', 'hippo_id', 'features', 'label']):
            logger.warning(f"Skipping behavior annotation entry: {ann_entry.get('annotation_id', 'Unknown ID')}"); continue
        training_data_list_for_df.append({
            'video_path': ann_entry['video_path'], 'frame_idx': ann_entry['sequence_end_original_frame'], 
            'timestamp_sec': ann_entry['timestamp_at_label_end'], 'hippo_id': ann_entry['hippo_id'], 
            'features': str(ann_entry['features']), 'label': ann_entry['label']
        })
    if not training_data_list_for_df: logger.warning("No valid behavior annotation entries to convert."); return False
    df = pd.DataFrame(training_data_list_for_df)
    try:
        df.to_csv(output_csv_file, index=False)
        logger.info(f"Converted behavior annotations to CSV: {output_csv_file}"); return True
    except Exception as e: logger.error(f"Error writing behavior training CSV: {e}"); return False


def run_inference_with_cnn_and_generate_insights(video_processor, cnn_detector, feature_extractor,
                                                 behavior_classifier, gemini_handler, video_files_for_inference):
    logger.info(f"Starting Inference with Custom CNN for {len(video_files_for_inference)} videos.")

    if not cnn_detector.loaded_cnn_model:
        load_cnn_detector_model() # Try to load it if not already
        if not cnn_detector.loaded_cnn_model:
            logger.error("Custom CNN detector model not loaded. Train it first (Option 1C)."); return

    # === DEFINE use_trained_behavior_classifier HERE ===
    use_trained_behavior_classifier = True # Default to True
    if not behavior_classifier.model or not behavior_classifier.label_encoder:
        logger.warning("Trained BehaviorClassifier model not loaded. Falling back to rule-based behaviors from cnn_hippo_detector.py or 'unknown'.")
        use_trained_behavior_classifier = False
    # ===================================================

    for video_file_path in video_files_for_inference:
        video_basename = os.path.basename(video_file_path)
        logger.info(f"\n--- CNN Inferring on Video: {video_basename} ---")
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video for inference: {video_file_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps == 0: fps = 25.0
        frame_pixel_area = frame_w * frame_h

        # Initialize bg_subtractor and prev_gray for each video
        bg_sub_infer = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY_CNN,
            varThreshold=config.MOG2_VAR_THRESHOLD_CNN,
            detectShadows=config.MOG2_DETECT_SHADOWS_CNN
        )
        prev_gray_infer = None
        
        current_hippo_states = {
            hid: {"bbox": None, "prev_bbox": None, "class_idx": prof["class_idx"], "feature_buffer": []}
            for hid, prof in config.HIPPO_PROFILES_CNN.items()
        }
        frame_num = 0
        daily_activities_summary_for_video = {hid: [] for hid in config.HIPPO_PROFILES_CNN.keys()}

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1
            timestamp = frame_num / fps

            for hid_prof in current_hippo_states:
                current_hippo_states[hid_prof]["prev_bbox"] = current_hippo_states[hid_prof]["bbox"]
                current_hippo_states[hid_prof]["bbox"] = None # Reset current bbox

            # Use instance methods for motion and CNN detection
            motion_proposals, current_gray_infer = cnn_detector.run_motion_detector_cnn(frame, prev_gray_infer, bg_sub_infer)
            prev_gray_infer = current_gray_infer
            cnn_detections_this_frame = cnn_detector.run_cnn_on_proposals(frame, motion_proposals)

            for hid_prof_id in sorted(config.HIPPO_PROFILES_CNN.keys()):
                best_det_for_hippo = None
                if cnn_detections_this_frame.get(hid_prof_id) and cnn_detections_this_frame[hid_prof_id]:
                    best_det_for_hippo = max(cnn_detections_this_frame[hid_prof_id], key=lambda d: d['confidence'])

                current_bbox_feature_for_bh = np.full(config.BBOX_FEATURE_LENGTH, np.nan)
                if best_det_for_hippo:
                    current_hippo_states[hid_prof_id]["bbox"] = best_det_for_hippo['bbox']
                    normalized_bbox_data = feature_extractor.normalize_bbox_features(
                        list(best_det_for_hippo['bbox']) + [best_det_for_hippo['confidence']],
                        frame_w, frame_h
                    )
                    current_bbox_feature_for_bh = feature_extractor.bbox_to_feature_vector(normalized_bbox_data)
                
                current_hippo_states[hid_prof_id]["feature_buffer"].append(current_bbox_feature_for_bh)
                if len(current_hippo_states[hid_prof_id]["feature_buffer"]) > config.SEQUENCE_LENGTH:
                    current_hippo_states[hid_prof_id]["feature_buffer"].pop(0)

            # Behavior Inference
            inferred_behaviors, inferred_emotions = {}, {}
            if use_trained_behavior_classifier: # Now defined
                for hid_bh, state_bh in current_hippo_states.items():
                    if len(state_bh["feature_buffer"]) == config.SEQUENCE_LENGTH:
                        buffer_to_predict = [np.array(fb_item) if not isinstance(fb_item, np.ndarray) else fb_item
                                             for fb_item in state_bh["feature_buffer"]]
                        # Ensure no empty arrays in buffer_to_predict before attempting reshape/flatten
                        valid_buffer_items = [item for item in buffer_to_predict if item.size > 0]
                        if len(valid_buffer_items) == config.SEQUENCE_LENGTH:
                            try:
                                sequence_to_predict = np.array(valid_buffer_items).flatten().reshape(1, -1)
                                if not np.all(np.isnan(sequence_to_predict)):
                                    predicted_label_list = behavior_classifier.predict(sequence_to_predict)
                                    inferred_behaviors[hid_bh] = predicted_label_list[0] if predicted_label_list and predicted_label_list[0] not in ["model_not_loaded", "input_error", "prediction_error", "imputation_error"] else "unknown_ml_pred"
                                    inferred_emotions[hid_bh] = "Neutral" # Placeholder
                                else:
                                    inferred_behaviors[hid_bh] = "out_of_frame" # if all features are NaN
                                    inferred_emotions[hid_bh] = "Neutral"
                            except ValueError as ve_reshape: # Catch potential reshape errors if flatten results in unexpected shape
                                logger.debug(f"Could not reshape feature sequence for hippo {hid_bh}: {ve_reshape}. Buffer: {valid_buffer_items}")
                                inferred_behaviors[hid_bh] = "unknown_seq_error"
                                inferred_emotions[hid_bh] = "Neutral"
                        else: # Not enough valid items in buffer for a full sequence
                            inferred_behaviors[hid_bh] = "unknown_short_buffer"
                            inferred_emotions[hid_bh] = "Neutral"

                    else: # Not enough data points in buffer for a sequence
                        inferred_behaviors[hid_bh] = "unknown_seq_too_short"
                        inferred_emotions[hid_bh] = "Neutral"
            else: # Rule-based
                # Pass current_hippo_states to the rule-based inferencer
                inferred_behaviors_rule, inferred_emotions_rule = infer_behavior_and_emotion_cnn(current_hippo_states, frame_pixel_area)
                inferred_behaviors.update(inferred_behaviors_rule)
                inferred_emotions.update(inferred_emotions_rule)

            for hid_log in config.HIPPO_PROFILES_CNN.keys():
                behavior = inferred_behaviors.get(hid_log, "out_of_frame" if current_hippo_states[hid_log]['bbox'] is None else "unknown")
                emotion = inferred_emotions.get(hid_log, "Neutral")
                if current_hippo_states[hid_log]['bbox'] is not None or behavior not in ["out_of_frame", "unknown", "unknown_seq_error", "unknown_short_buffer", "unknown_seq_too_short"]:
                     logger.info(f"  F:{frame_num} T:{timestamp:.2f}s Hippo {hid_log}: Det={current_hippo_states[hid_log]['bbox'] is not None}, Beh:{behavior}, Emo:{emotion}")
                     daily_activities_summary_for_video[hid_log].append({'time': timestamp, 'behavior': behavior, 'emotion': emotion})
        
        cap.release()
        if gemini_handler and gemini_handler.model:
            for hippo_id_summary, activities_list in daily_activities_summary_for_video.items():
                if activities_list:
                    activities_list.sort(key=lambda x: x['time'])
                    summary_str_parts = []
                    last_behavior_sum = None; start_time_sum = None; count_sum = 0; emotion_list_sum = []
                    for act_idx, act in enumerate(activities_list):
                        if act['behavior'] != last_behavior_sum or act_idx == 0 : # Start new summary part
                            if last_behavior_sum and start_time_sum is not None:
                                duration_sum = activities_list[act_idx-1]['time'] - start_time_sum # Duration of previous behavior block
                                avg_emotion = max(set(emotion_list_sum), key=emotion_list_sum.count) if emotion_list_sum else "Neutral"
                                summary_str_parts.append(f"{last_behavior_sum} (~{duration_sum:.0f}s, {count_sum} instances, feeling {avg_emotion})")
                            last_behavior_sum = act['behavior']; start_time_sum = act['time']; count_sum = 1; emotion_list_sum = [act['emotion']]
                        else: 
                            count_sum +=1
                            emotion_list_sum.append(act['emotion'])
                    
                    # Add the last ongoing activity block
                    if last_behavior_sum and start_time_sum is not None:
                        duration_sum = activities_list[-1]['time'] - start_time_sum # Duration up to last recorded event
                        avg_emotion = max(set(emotion_list_sum), key=emotion_list_sum.count) if emotion_list_sum else "Neutral"
                        summary_str_parts.append(f"{last_behavior_sum} (~{duration_sum:.0f}s, {count_sum} instances, feeling {avg_emotion})")

                    summary_str_final = "; ".join(summary_str_parts[:10]) # Limit number of segments in summary
                    if len(summary_str_parts) > 10: summary_str_final += " ...and more activities."
                    if not summary_str_final: summary_str_final = "various activities observed."
                    
                    digest = gemini_handler.generate_daily_digest(
                        f"Hippo_{hippo_id_summary}_from_{video_basename}",
                        summary_str_final,
                        persona_name=config.HIPPO_PROFILES_CNN.get(hippo_id_summary, {}).get("name", f"Hippo {hippo_id_summary}")
                    )
                    logger.info(f"\n  --- Daily Digest for Hippo {hippo_id_summary} (Video: {video_basename}) ---\n  {digest}\n")

    logger.info("CNN Inference and insights generation finished.")


def main():
    ensure_dotenv_at_root()
    video_proc = VideoProcessor()
    cnn_detector = CustomCNNDetector() # Uses custom CNN
    feat_ext = FeatureExtractor()
    behav_class = BehaviorClassifier() # Uses bbox features
    gemini_hand = GeminiHandler()

    # Initial checks
    if not os.path.exists(config.DETECTION_MODEL_ONNX_VP):
        logger.critical(f"Initial VideoProcessor detection model not found: {config.DETECTION_MODEL_ONNX_VP}. Clip generation (1A/3A) will fail.")
        # return # Or allow user to proceed if they know what they are doing

    # Load trained CNN model if it exists, so other options can check for it
    load_cnn_detector_model() # This function is from cnn_hippo_detector module

    all_video_files_in_data_dir = video_proc.scan_video_folders(config.DATA_DIR)
    # ... (train/test split logic from your previous main.py) ...
    train_video_files = []; test_video_files = []
    if not all_video_files_in_data_dir: logger.warning(f"No videos in DATA_DIR: {config.DATA_DIR}.")
    test_folder_full_path = os.path.join(config.DATA_DIR, config.TEST_VIDEO_FOLDER_NAME)
    if config.TEST_VIDEO_FOLDER_NAME and os.path.isdir(test_folder_full_path):
        for vid_path in all_video_files_in_data_dir:
            if os.path.commonpath([vid_path, test_folder_full_path]) == os.path.normpath(test_folder_full_path):
                test_video_files.append(vid_path)
            else: train_video_files.append(vid_path)
    else: train_video_files = all_video_files_in_data_dir
    logger.info(f"Found {len(train_video_files)} videos for TRAIN pool.")
    logger.info(f"Found {len(test_video_files)} videos for TEST pool.")


    while True:
        print("\nHippoSphere AI - Custom CNN Detector Workflow:")
        print("--- Setup & Training for Custom CNN Detector ---")
        print("1A. Generate CLIPS from TRAIN set (Input for CNN BBox Annotation)")
        print("1B. Annotate BBoxes on CLIPS (To create training data for CNN detector)")
        print("1C. Train Custom CNN Hippo Detector (Using annotated bboxes/patches)")
        print("--- Behavior Analysis Pipeline (Uses Trained CNN Detector) ---")
        print("2A. Process TRAIN Clips: CNN Detect -> Extract BBox Features")
        print("2B. Annotate BEHAVIORS on TRAIN Features (Using AnnotationTool.py)")
        print("2C. Train BEHAVIOR Classifier (Using features from 2A & labels from 2B)")
        print("--- Testing & Inference ---")
        print("3A. Generate CLIPS from TEST set")
        print("3B. Process TEST Clips: CNN Detect -> Extract BBox Features")
        print("3C. Run Inference & Insights on TEST set (Using trained CNN & Behavior Model)")
        print("3D. Run Inference & Insights on TRAIN set")
        print("4.  EXIT")
        choice = input("Enter your choice: ").strip().upper()

        if choice == '1A':
            if not train_video_files: logger.error("No TRAIN video files for clip generation.")
            else: generate_clips_pipeline(video_proc, train_video_files)
        
        elif choice == '1B':
            logger.info("Starting Bounding Box Annotation for CNN Training...")
            run_cnn_bbox_annotation_mode()
            logger.info(f"CNN BBox Annotation finished. Check: {config.CNN_ANNOTATIONS_FILE}")

        elif choice == '1C':
            logger.info("Starting Custom CNN Hippo Detector Training...")
            train_custom_cnn_detector_model()
            logger.info(f"CNN training finished. Model: {config.CNN_MODEL_SAVE_PATH}")

        elif choice == '2A':
            logger.info("Processing TRAIN Clips: CNN Detections & Feature Extraction...")
            if not train_video_files: 
                logger.error("No TRAIN video files to find clips from.")
            # === CORRECTED CHECK: Access the instance's model attribute ===
            elif not cnn_detector.loaded_cnn_model:
                logger.error("Custom CNN model not loaded in detector instance. Train it first (Option 1C) or check loading logs.")
            # ============================================================
            else: 
                cnn_detection_and_feature_pipeline(cnn_detector, feat_ext, train_video_files)
        
        elif choice == '2B':
            logger.info("Starting Behavior Annotation...")
            json_behavior_annotation_file = os.path.join(config.BEHAVIOR_ANNOTATIONS_DIR, "hippo_behavior_annotations.json")
            annot_tool = AnnotationTool(features_dir=config.DETECTIONS_DATA_DIR,
                                        clips_dir=config.CLIPS_DIR,
                                        output_annotation_file=json_behavior_annotation_file)
            annot_tool.annotate_segment_features()
            csv_behavior_training_file = os.path.join(config.BEHAVIOR_ANNOTATIONS_DIR, "behavior_training_data_from_bbox.csv")
            if convert_behavior_annotations_to_training_csv(json_behavior_annotation_file, csv_behavior_training_file):
                logger.info(f"Behavior Training CSV ready: {csv_behavior_training_file}.")

        elif choice == '2C':
            logger.info("Starting Behavior Classifier Training...")
            csv_behavior_training_file = os.path.join(config.BEHAVIOR_ANNOTATIONS_DIR, "behavior_training_data_from_bbox.csv")
            if not os.path.exists(csv_behavior_training_file):
                logger.error(f"Behavior Training CSV not found: {csv_behavior_training_file}. Run 2B first.")
            else: behav_class.train(csv_behavior_training_file)

        elif choice == '3A':
            if not test_video_files: logger.error("No TEST video files for clip generation.")
            else: generate_clips_pipeline(video_proc, test_video_files)

        elif choice == '3B':
                logger.info("Processing TEST Clips for BBox Features...")
                if not test_video_files: logger.error("No TEST video files to find clips from.")
                # === CORRECTED CHECK ===
                elif not cnn_detector.loaded_cnn_model:
                    logger.error("Custom CNN model not loaded in detector instance. Train it first (1C) or check loading logs.")
                # =======================
                else: cnn_detection_and_feature_pipeline(cnn_detector, feat_ext, test_video_files)

        elif choice == '3C':
            logger.info("Running Inference on TEST videos/clips...")
            if not test_video_files: logger.error("No TEST video files for inference.")
            else: run_inference_with_cnn_and_generate_insights(video_proc, cnn_detector, feat_ext, behav_class, gemini_hand, test_video_files)
        
        elif choice == '3D':
            logger.info("Running Inference on TRAIN videos/clips...")
            if not train_video_files: logger.error("No TRAIN video files for inference.")
            else: run_inference_with_cnn_and_generate_insights(video_proc, cnn_detector, feat_ext, behav_class, gemini_hand, train_video_files)

        elif choice == '4':
            logger.info("Exiting HippoSphere AI. Goodbye!")
            break
        else:
            logger.warning("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()