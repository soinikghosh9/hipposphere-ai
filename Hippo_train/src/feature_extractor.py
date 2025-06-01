# src/feature_extractor.py
import numpy as np
import logging
from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO), # Use getattr for safety
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Use the length defined in config for consistency
if not hasattr(config, 'BBOX_FEATURE_LENGTH'):
    logger.critical("CRITICAL: config.BBOX_FEATURE_LENGTH is not defined. FeatureExtractor may not work correctly.")
    # Define a fallback, but this indicates a setup issue
    EXPECTED_BBOX_FEATURE_LENGTH = 5
else:
    EXPECTED_BBOX_FEATURE_LENGTH = config.BBOX_FEATURE_LENGTH


class FeatureExtractor:
    def __init__(self):
        pass

    def normalize_bbox_features(self, bbox_data_raw, frame_width, frame_height):
        """
        Normalizes bounding box features.
        Input: bbox_data_raw = [x, y, w, h, confidence] or list of NaNs
        Output: normalized features [center_x_norm, center_y_norm, w_norm, h_norm, confidence]
        Coordinates and dimensions are normalized by frame width/height.
        """
        if not isinstance(bbox_data_raw, list) or len(bbox_data_raw) != EXPECTED_BBOX_FEATURE_LENGTH:
            # Check if all elements are NaN if length is correct but values are bad
            if isinstance(bbox_data_raw, list) and len(bbox_data_raw) == EXPECTED_BBOX_FEATURE_LENGTH and np.all(np.isnan(bbox_data_raw)):
                return np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan).tolist()
            logger.debug(f"Invalid bbox_data_raw for normalization. Expected list of length {EXPECTED_BBOX_FEATURE_LENGTH}. Got: {bbox_data_raw}")
            return np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan).tolist()

        # If all are NaN (already checked basically, but an explicit pass)
        if np.all(np.isnan(bbox_data_raw)):
            return np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan).tolist()

        x, y, w, h, conf = bbox_data_raw

        # Check if coordinate/dimension data is valid before division
        if any(v is None or np.isnan(v) for v in [x, y, w, h]): # Check first 4 elements for NaN
             logger.debug(f"NaN found in bbox coordinates/dimensions: {[x,y,w,h]}. Returning NaNs.")
             return np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan).tolist()

        if frame_width is None or frame_height is None or frame_width == 0 or frame_height == 0:
            logger.warning("Frame width or height is invalid for bbox normalization. Returning NaNs.")
            return np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan).tolist()

        center_x_norm = (float(x) + float(w) / 2.0) / float(frame_width)
        center_y_norm = (float(y) + float(h) / 2.0) / float(frame_height)
        w_norm = float(w) / float(frame_width)
        h_norm = float(h) / float(frame_height)
        
        return [center_x_norm, center_y_norm, w_norm, h_norm, float(conf) if not np.isnan(conf) else np.nan]

    def bbox_to_feature_vector(self, normalized_bbox_features):
        """
        Converts normalized bounding box features to a flat feature vector.
        Input: [center_x_norm, center_y_norm, w_norm, h_norm, confidence]
        Output: A flat NumPy array.
        """
        if not isinstance(normalized_bbox_features, list) or len(normalized_bbox_features) != EXPECTED_BBOX_FEATURE_LENGTH:
            return np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan)
        
        return np.array(normalized_bbox_features, dtype=float)

    def process_segment_data_for_features(self, segment_detections_data_list, frame_width=None, frame_height=None):
        """
        Processes a list of detection data entries for a segment, normalizes bboxes,
        and extracts feature vectors.
        """
        segment_features_output = []

        if frame_width is None or frame_height is None or frame_width <=0 or frame_height <=0 :
            logger.warning("Frame dimensions not provided or invalid for FeatureExtractor. "
                           "Attempting to use defaults from config.DETECTION_INPUT_WIDTH_VP, which might be inaccurate.")
            # These VP dimensions are for the initial YOLOv5s, not necessarily the clip dimensions.
            # It's CRITICAL that main.py passes the correct dimensions of the clip being processed.
            frame_width_used = config.DETECTION_INPUT_WIDTH_VP
            frame_height_used = config.DETECTION_INPUT_HEIGHT_VP
            if frame_width_used <= 0 or frame_height_used <= 0:
                logger.error("Fallback frame dimensions from config are also invalid. Cannot normalize features.")
                # Populate with NaNs if dimensions are unusable
                for frame_data_entry in segment_detections_data_list:
                    num_hippos = len(config.HIPPO_PROFILES_CNN) # Number of expected feature vectors
                    feature_vectors_for_this_frame = [np.full(EXPECTED_BBOX_FEATURE_LENGTH, np.nan) for _ in range(num_hippos)]
                    segment_features_output.append({
                        'video_path': frame_data_entry.get('video_path', 'unknown_video'),
                        'frame_idx': frame_data_entry.get('frame_idx', -1),
                        'timestamp_sec': frame_data_entry.get('timestamp_sec', -1.0),
                        'feature_vectors': feature_vectors_for_this_frame
                    })
                return segment_features_output
        else:
            frame_width_used = frame_width
            frame_height_used = frame_height


        for frame_data_entry in segment_detections_data_list:
            # 'hippo_bboxes_conf' is a list of lists, one per hippo profile.
            # Each inner list is [x,y,w,h,conf] or NaNs.
            raw_bboxes_for_frame = frame_data_entry.get('hippo_bboxes_conf', [])
            
            feature_vectors_for_this_frame = []
            for single_hippo_bbox_data in raw_bboxes_for_frame:
                normalized_bbox = self.normalize_bbox_features(single_hippo_bbox_data, frame_width_used, frame_height_used)
                fv = self.bbox_to_feature_vector(normalized_bbox)
                feature_vectors_for_this_frame.append(fv)

            segment_features_output.append({
                'video_path': frame_data_entry.get('video_path', 'unknown_video'), # Added .get for safety
                'frame_idx': frame_data_entry.get('frame_idx', -1),
                'timestamp_sec': frame_data_entry.get('timestamp_sec', -1.0),
                'feature_vectors': feature_vectors_for_this_frame
            })
        return segment_features_output

if __name__ == '__main__':
    fe = FeatureExtractor()
    frame_w_test, frame_h_test = 640, 480
    
    # Test with a hippo profile structure similar to config
    # This assumes config.HIPPO_PROFILES_CNN is defined for the test to infer num_hippos
    num_test_hippos = len(config.HIPPO_PROFILES_CNN) if hasattr(config, 'HIPPO_PROFILES_CNN') and config.HIPPO_PROFILES_CNN else 2

    raw_detections_one_frame = []
    # Hippo 1 detected
    raw_detections_one_frame.append([100, 150, 50, 80, 0.95])
    # Hippo 2 not detected (assuming 2 profiles for test if config.HIPPO_PROFILES_CNN is not set)
    if num_test_hippos > 1:
        raw_detections_one_frame.append([np.nan] * EXPECTED_BBOX_FEATURE_LENGTH) 
    # Add more NaNs if num_test_hippos > 2
    for _ in range(2, num_test_hippos):
        raw_detections_one_frame.append([np.nan] * EXPECTED_BBOX_FEATURE_LENGTH)


    dummy_segment_detections_data = [{
        'video_path': 'test.mp4',
        'frame_idx': 0,
        'timestamp_sec': 0.0,
        'hippo_bboxes_conf': raw_detections_one_frame
    }]

    logger.info("--- Testing FeatureExtractor with Bounding Box Data ---")
    features_output = fe.process_segment_data_for_features(dummy_segment_detections_data, frame_w_test, frame_h_test)
    
    if features_output:
        logger.info(f"Processed {len(features_output)} frames.")
        first_frame_features = features_output[0]['feature_vectors']
        if len(first_frame_features) > 0:
            logger.info(f"Feature vector for Hippo Profile 0 (normalized bbox): {first_frame_features[0]}")
            assert len(first_frame_features[0]) == EXPECTED_BBOX_FEATURE_LENGTH
        if len(first_frame_features) > 1:
            logger.info(f"Feature vector for Hippo Profile 1 (NaNs): {first_frame_features[1]}")
            assert np.all(np.isnan(first_frame_features[1]))
    else:
        logger.error("Feature extraction test failed.")