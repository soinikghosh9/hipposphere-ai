# src/video_processor.py
import cv2
import os
import numpy as np
from tqdm import tqdm
import logging
import onnxruntime as ort

from src import config # Uses config variables with _VP suffix now

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.ort_session = None
        self.input_name = None
        self.output_names = None
        self._load_detector_ort()

        if self.ort_session is None:
            logger.error("Failed to load object detector with ONNXRuntime. Detection will fail.")

        self.class_names = []
        # === UPDATED TO USE _VP SUFFIX ===
        if hasattr(config, 'DETECTION_CLASSES_FILE_VP') and config.DETECTION_CLASSES_FILE_VP and \
           os.path.exists(config.DETECTION_CLASSES_FILE_VP):
            try:
                with open(config.DETECTION_CLASSES_FILE_VP, 'rt') as f:
                    self.class_names = f.read().rstrip('\n').split('\n')
                logger.info(f"Loaded {len(self.class_names)} class names from {config.DETECTION_CLASSES_FILE_VP}.")
            except Exception as e:
                logger.error(f"Error reading classes file {config.DETECTION_CLASSES_FILE_VP}: {e}")
        else:
            logger.warning(f"Classes file for VideoProcessor not found or not configured: config.DETECTION_CLASSES_FILE_VP.")
        # ===================================

    def _load_detector_ort(self):
        try:
            # === UPDATED TO USE _VP SUFFIX ===
            if hasattr(config, 'DETECTION_MODEL_ONNX_VP') and config.DETECTION_MODEL_ONNX_VP and \
               os.path.exists(config.DETECTION_MODEL_ONNX_VP):
                logger.info(f"Loading ONNX detection model with ONNXRuntime from: {config.DETECTION_MODEL_ONNX_VP}")
            # ===================================
                providers = []
                if ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')

                self.ort_session = ort.InferenceSession(config.DETECTION_MODEL_ONNX_VP, providers=providers)
                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_names = [output.name for output in self.ort_session.get_outputs()]
                
                logger.info(f"ONNXRuntime session created for '{config.DETECTION_MODEL_ONNX_VP}'")
                logger.info(f"  Input Name: {self.input_name}, Shape: {self.ort_session.get_inputs()[0].shape}")
                logger.info(f"  Output Names: {self.output_names}")
                for output_node in self.ort_session.get_outputs(): # Renamed variable to avoid conflict
                    logger.info(f"    Output '{output_node.name}' Shape: {output_node.shape}")
            else:
                logger.error(f"Detection model ONNX file for VideoProcessor not found or not configured: config.DETECTION_MODEL_ONNX_VP")
        except Exception as e:
            logger.error(f"Error loading object detector with ONNXRuntime: {e}", exc_info=True)
            self.ort_session = None


    def _preprocess_frame_for_ort(self, frame):
        # === UPDATED TO USE _VP SUFFIX for input dimensions ===
        img_resized = cv2.resize(frame, (config.DETECTION_INPUT_WIDTH_VP, config.DETECTION_INPUT_HEIGHT_VP))
        # ======================================================
        img_chw = img_resized.transpose(2, 0, 1)
        img_normalized = img_chw / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0).astype(np.float16)
        return input_tensor

    def _postprocess_ort_yolo_output(self, ort_outs, frame_width, frame_height):
        boxes = []
        confidences = []
        class_ids_raw = []

        detections_tensor = ort_outs[0]
        
        # === UPDATED TO USE _VP SUFFIX for thresholds and input dimensions ===
        expected_output_dim = 4 + 1 + len(self.class_names) if self.class_names else 85 # Fallback if class_names not loaded
        if detections_tensor.shape[2] == expected_output_dim : # Check if last dimension matches expectation
        # =======================================================================
            num_predictions = detections_tensor.shape[1]
            predictions = detections_tensor[0]

            for i in range(num_predictions):
                prediction = predictions[i]
                object_confidence = prediction[4]

                # === UPDATED TO USE _VP SUFFIX ===
                if object_confidence > config.DETECTION_CONF_THRESHOLD_VP:
                # ===================================
                    class_scores = prediction[5:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id] * object_confidence

                    # === UPDATED TO USE _VP SUFFIX ===
                    if confidence > config.DETECTION_CONF_THRESHOLD_VP:
                    # ===================================
                        cx_net, cy_net, w_net, h_net = prediction[0:4]

                        # === UPDATED TO USE _VP SUFFIX ===
                        scale_x_to_orig = frame_width / config.DETECTION_INPUT_WIDTH_VP
                        scale_y_to_orig = frame_height / config.DETECTION_INPUT_HEIGHT_VP
                        # ===================================

                        x_orig = int((cx_net - w_net / 2) * scale_x_to_orig)
                        y_orig = int((cy_net - h_net / 2) * scale_y_to_orig)
                        w_f_orig = int(w_net * scale_x_to_orig)
                        h_f_orig = int(h_net * scale_y_to_orig)
                        
                        x_orig = max(0, x_orig)
                        y_orig = max(0, y_orig)
                        w_f_orig = min(w_f_orig, frame_width - x_orig)
                        h_f_orig = min(h_f_orig, frame_height - y_orig)

                        # === UPDATED TO USE _VP SUFFIX ===
                        min_width_px = frame_width * config.MIN_OBJECT_WIDTH_PERCENT_VP
                        min_height_px = frame_height * config.MIN_OBJECT_HEIGHT_PERCENT_VP
                        # ===================================

                        if w_f_orig >= min_width_px and h_f_orig >= min_height_px and w_f_orig > 0 and h_f_orig > 0:
                            boxes.append([x_orig, y_orig, w_f_orig, h_f_orig])
                            confidences.append(float(confidence))
                            class_ids_raw.append(class_id)
        else:
            logger.warning(f"Unexpected ONNX output shape for VideoProcessor YOLO: {detections_tensor.shape}. "
                           f"Expected last dim {expected_output_dim} (4+1+num_classes). Got {detections_tensor.shape[2]}. Cannot parse detections.")

        indices = []
        if boxes:
            # === UPDATED TO USE _VP SUFFIX ===
            indices = cv2.dnn.NMSBoxes(boxes, confidences, config.DETECTION_CONF_THRESHOLD_VP, config.DETECTION_NMS_THRESHOLD_VP)
            # ===================================

        detected_regions = []
        if len(indices) > 0: # Check if indices is not empty before trying to flatten
            for i in indices.flatten():
                box = boxes[i]
                detected_regions.append({'box': box, 'confidence': confidences[i]})
        return detected_regions


    def detect_hippos_in_frame(self, frame):
        if self.ort_session is None:
            return []
        frame_height, frame_width = frame.shape[:2]
        input_tensor = self._preprocess_frame_for_ort(frame)
        try:
            ort_inputs = {self.input_name: input_tensor}
            ort_outs = self.ort_session.run(self.output_names, ort_inputs)
        except Exception as e:
            logger.error(f"Error during ONNXRuntime inference: {e}", exc_info=True)
            return []
        return self._postprocess_ort_yolo_output(ort_outs, frame_width, frame_height)
    
    def scan_video_folders(self, root_folder):
        video_files = []
        logger.info(f"Scanning for video files in root folder: {root_folder}")
        if not os.path.isdir(root_folder):
            logger.error(f"Provided DATA_DIR is not a valid directory: {root_folder}")
            return []
        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                # === UPDATED TO USE _VP SUFFIX (though VIDEO_EXTENSIONS is general) ===
                if filename.lower().endswith(config.VIDEO_EXTENSIONS): # VIDEO_EXTENSIONS doesn't have _VP
                # ========================================================================
                    video_files.append(os.path.join(dirpath, filename))
        logger.info(f"Found {len(video_files)} video files in {root_folder}.")
        return video_files

    def extract_active_segments(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or total_frames == 0:
            logger.warning(f"FPS is {fps} or Total Frames is {total_frames} for video {video_path}. Defaulting FPS to 25.")
            if total_frames == 0: cap.release(); return []
            fps = 25.0 # Ensure float for division

        logger.info(f"Processing video for segments: {os.path.basename(video_path)}, Total frames: {total_frames}, FPS: {fps:.2f}")

        active_segments = []
        current_segment_start_frame = -1
        frame_read_ok = True
        # === UPDATED TO USE _VP SUFFIX ===
        sample_rate = config.FRAME_SAMPLE_RATE_DETECTION_VP
        # ===================================
        if sample_rate <= 0: sample_rate = 1

        prev_gray = None 
        motion_active_for_segment = False 

        for frame_idx in tqdm(range(total_frames), desc=f"Scanning {os.path.basename(video_path)}"):
            if frame_idx % sample_rate == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} from {video_path}.")
                    frame_read_ok = False
                else:
                    frame_read_ok = True

                object_present_in_frame = False
                if frame_read_ok:
                    detected_regions = self.detect_hippos_in_frame(frame)
                    object_present_in_frame = len(detected_regions) > 0

                motion_in_this_frame = False
                if object_present_in_frame and frame_read_ok:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0) # Standard Gaussian blur kernel
                    if prev_gray is not None:
                        frame_diff = cv2.absdiff(prev_gray, gray)
                        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY) # Standard thresholding
                        motion_score = np.sum(thresh > 0)
                        # === UPDATED TO USE _VP SUFFIX ===
                        if motion_score > config.MOTION_THRESHOLD_VP:
                        # ===================================
                            motion_in_this_frame = True
                            motion_active_for_segment = True
                    prev_gray = gray
                
                is_active_frame = object_present_in_frame and motion_in_this_frame

                if is_active_frame and frame_read_ok:
                    if current_segment_start_frame == -1:
                        current_segment_start_frame = frame_idx
                        motion_active_for_segment = motion_in_this_frame 
                else:
                    if current_segment_start_frame != -1:
                        # === UPDATED TO USE _VP SUFFIX ===
                        if not motion_active_for_segment and config.MOTION_THRESHOLD_VP > 0: # Check if motion was required
                        # ===================================
                            pass # logger.debug(f"Segment discarded (no motion).")
                        else:
                            segment_duration_frames = (frame_idx - 1) - current_segment_start_frame
                            segment_duration_seconds = segment_duration_frames / fps
                            # === UPDATED TO USE _VP SUFFIX ===
                            if segment_duration_seconds >= config.MIN_SEGMENT_DURATION_SECONDS_VP:
                            # ===================================
                                active_segments.append({
                                    'video_path': video_path,
                                    'start_frame': current_segment_start_frame,
                                    'end_frame': frame_idx - 1,
                                    'fps': fps
                                })
                        current_segment_start_frame = -1
                        motion_active_for_segment = False 
                    if not frame_read_ok:
                        break
            
        if current_segment_start_frame != -1:
            # === UPDATED TO USE _VP SUFFIX (Optional check, if motion threshold > 0) ===
            if motion_active_for_segment or config.MOTION_THRESHOLD_VP == 0:
            # =======================================================================
                segment_duration_frames = (total_frames - 1) - current_segment_start_frame
                segment_duration_seconds = segment_duration_frames / fps
                # === UPDATED TO USE _VP SUFFIX ===
                if segment_duration_seconds >= config.MIN_SEGMENT_DURATION_SECONDS_VP:
                # ===================================
                    active_segments.append({
                        'video_path': video_path,
                        'start_frame': current_segment_start_frame,
                        'end_frame': total_frames - 1,
                        'fps': fps
                    })

        cap.release()
        logger.info(f"Found {len(active_segments)} active segments in {os.path.basename(video_path)}.")
        return active_segments

    def extract_clip(self, video_path, start_frame, end_frame, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file for clip extraction: {video_path}")
            return False
        # Use a general FPS for clips, or original video's FPS
        fps_clip = cap.get(cv2.CAP_PROP_FPS)
        # FPS_FOR_POSE_ESTIMATION is for the *frames processed by pose estimator*, not necessarily clip FPS
        # If FPS_FOR_POSE_ESTIMATION is intended for clip FPS, ensure it's correctly named or used.
        # For now, using original video FPS for clips is safer.
        if fps_clip == 0: fps_clip = 25.0 # Fallback

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps_clip, (width, height))
        if not out_writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {output_path}.")
            cap.release(); return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx_iter in range(start_frame, end_frame + 1):
            ret, frame_iter = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {frame_idx_iter} while extracting clip. Ending early.")
                break
            out_writer.write(frame_iter)
        cap.release()
        out_writer.release()
        logger.info(f"Successfully extracted clip: {os.path.basename(output_path)}")
        return True

# __main__ block for testing (can be kept or simplified)
if __name__ == '__main__':
    if not os.path.exists(".env"):
        with open(".env", "w") as f: f.write("GEMINI_API_KEY=YOUR_API_KEY_HERE\n")

    logger.info("Testing VideoProcessor with ONNXRuntime integration...")
    processor_ort = VideoProcessor()

    if processor_ort.ort_session is None:
        logger.error("VideoProcessor's ONNXRuntime session failed to load. Cannot run example usage effectively.")
        logger.error(f"Ensure '{config.DETECTION_MODEL_ONNX}' exists in '{config.MODELS_DIR}' and is a valid ONNX model.")
    else:
        dummy_video_path = os.path.join(config.DATA_DIR, "dummy_detection_test_video.mp4")
        if not os.path.exists(dummy_video_path):
            logger.info("Creating a dummy video for video_processor.py (ORT) testing...")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out_dummy = cv2.VideoWriter(dummy_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 480))
            for i in range(25 * 5): # 5 seconds
                x_pos = 50 + int(400 * (i / (25*5)))
                cv2.rectangle(frame, (x_pos, 100), (x_pos + 150, 250), (255,255,255), -1) # Larger white rectangle
                out_dummy.write(frame)
                frame[100:250, x_pos:x_pos+150] = 0
            out_dummy.release()
            logger.info(f"Dummy video created at {dummy_video_path}")

        video_files = processor_ort.scan_video_folders(config.DATA_DIR)
        if video_files:
            test_video = video_files[0] # Use the first video found (could be the dummy)
            logger.info(f"Attempting to extract active segments from: {os.path.basename(test_video)}")
            segments = processor_ort.extract_active_segments(test_video)
            if segments:
                logger.info(f"Found {len(segments)} segments. Details of first: {segments[0]}")
                first_s = segments[0]
                clip_fn = f"{os.path.splitext(os.path.basename(first_s['video_path']))[0]}_ORT_segment_{first_s['start_frame']}_{first_s['end_frame']}.mp4"
                clip_out = os.path.join(config.CLIPS_DIR, clip_fn)
                processor_ort.extract_clip(first_s['video_path'], first_s['start_frame'], first_s['end_frame'], clip_out)
            else:
                logger.info(f"No active segments found in {os.path.basename(test_video)} using ORT.")
        else:
            logger.warning(f"No video files found in {config.DATA_DIR} to test with ORT.")
    logger.info("VideoProcessor (ORT) __main__ test finished.")