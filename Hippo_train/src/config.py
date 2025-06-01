# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = r"C:\Users\User\Downloads\20250109-20250601T082948Z-1-002\20250109"# YOUR ACTUAL PATH

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
CLIPS_DIR = os.path.join(PROCESSED_DATA_DIR, "clips")

CNN_ANNOTATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, "annotations", "hippo_cnn_bbox_annotations.json")
CNN_PATCHES_DIR = os.path.join(PROCESSED_DATA_DIR, "cnn_patches/")
HIPPO1_PATCH_DIR_CNN = os.path.join(CNN_PATCHES_DIR, "hippo1_large")
HIPPO2_PATCH_DIR_CNN = os.path.join(CNN_PATCHES_DIR, "hippo2_small")
BACKGROUND_PATCHES_DIR_CNN = os.path.join(CNN_PATCHES_DIR, "background")

DETECTIONS_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "detections_and_features")
BEHAVIOR_ANNOTATIONS_DIR = os.path.join(PROCESSED_DATA_DIR, "behavior_annotations")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Ensure directories exist (simplified) ---
for path in [DATA_DIR, PROCESSED_DATA_DIR, CLIPS_DIR, CNN_PATCHES_DIR,
             HIPPO1_PATCH_DIR_CNN, HIPPO2_PATCH_DIR_CNN, BACKGROUND_PATCHES_DIR_CNN,
             DETECTIONS_DATA_DIR, BEHAVIOR_ANNOTATIONS_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)

# --- Video Processing Parameters ---
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
FRAME_SAMPLE_RATE_DETECTION_VP = 125
MIN_SEGMENT_DURATION_SECONDS_VP = 10
MOTION_THRESHOLD_VP = 1000

# --- Initial Region Detection ---
DETECTION_MODEL_ONNX_VP = os.path.join(MODELS_DIR, "yolov5s.onnx")
DETECTION_CLASSES_FILE_VP = os.path.join(MODELS_DIR, "coco.names")
DETECTION_CLASS_FILTER_VP = ""
DETECTION_CONF_THRESHOLD_VP = 0.25
DETECTION_NMS_THRESHOLD_VP = 0.45
DETECTION_INPUT_WIDTH_VP = 640
DETECTION_INPUT_HEIGHT_VP = 640
MIN_OBJECT_WIDTH_PERCENT_VP = 0.05
MIN_OBJECT_HEIGHT_PERCENT_VP = 0.05

# --- Custom CNN Hippo Detector Parameters ---
CNN_MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "hippo_detector_cnn.h5")
CNN_IMG_WIDTH, CNN_IMG_HEIGHT = 64, 64
CNN_NUM_CLASSES = 3
CNN_BACKGROUND_CLASS_IDX = CNN_NUM_CLASSES - 1
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 30
CNN_CONFIDENCE_THRESHOLD = 0.75
BBOX_FEATURE_LENGTH = 5

# --- Tracking & Motion Detection Parameters ---
GAUSSIAN_BLUR_KERNEL_SIZE_CNN=(5,5); FRAME_DIFF_THRESHOLD_CNN=18
MOG2_HISTORY_CNN=500; MOG2_VAR_THRESHOLD_CNN=25; MOG2_DETECT_SHADOWS_CNN=False
MORPH_OPEN_BG_KERNEL_SIZE_CNN=(3,3); MORPH_CLOSE_BG_KERNEL_SIZE_CNN=(17,17)
MORPH_OPEN_FD_KERNEL_SIZE_CNN=(3,3); MORPH_CLOSE_FD_KERNEL_SIZE_CNN=(15,15)
TRACKER_TYPE_CV_CNN="CSRT"

# --- Hippo Profiles ---
HIPPO_PROFILES_CNN = {
    1: {"name":"Hippo 1 (L)", "name_short": "H1L", # Added name_short for brevity on screen
        "min_area":700,"max_area":40000,"min_solidity":0.60,"min_aspect":0.4,"max_aspect":3.0,
        "patch_dir":HIPPO1_PATCH_DIR_CNN,
        "class_idx":0, 
        "feature_vector_index": 0, # ADD THIS: Hippo 1 (L) uses index 0 of feature_vectors
        "color": (0, 255, 0) # Green
       }, 
    2: {"name":"Hippo 2 (S)", "name_short": "H2S", # Added name_short
        "min_area":100,"max_area":6000,"min_solidity":0.55,"min_aspect":0.3,"max_aspect":2.8,
        "patch_dir":HIPPO2_PATCH_DIR_CNN,
        "class_idx":1, 
        "feature_vector_index": 1, # ADD THIS: Hippo 2 (S) uses index 1 of feature_vectors
        "color": (0, 0, 255)  # Red
       }  
}
BACKGROUND_ANNOTATION_COLOR_CNN = (0, 165, 255) # Orange

# --- Behavior & Emotion Classification ---
BEHAVIOR_MODEL_BBOX_PATH = os.path.join(MODELS_DIR, "hippo_behavior_classifier_bbox.joblib")
BEHAVIOR_LABEL_ENCODER_BBOX_PATH = os.path.join(MODELS_DIR, "hippo_behavior_label_encoder_bbox.joblib")
BEHAVIOR_IMPUTER_BBOX_PATH = os.path.join(MODELS_DIR, "hippo_behavior_imputer_bbox.joblib")

# === REDUCED BEHAVIOR CLASSES ===
BEHAVIOR_CLASSES = [
    "resting_or_sleeping",   # Key 1
    "feeding_or_grazing",    # Key 2
    "walking_or_pacing",     # Key 3
    "swimming_or_wallowing", # Key 4
    "social_interaction",    # Key 5 (can be positive/negative, or a general interaction)
    "other_active"           # Key 6 (includes vigilance, playing, object interaction if not distinct)
    # "out_of_frame" is handled by lack of detection
]
# ================================

# === REDUCED EMOTION CLASSES (for manual annotation) ===
# If you don't want to manually annotate emotions, set this to None or empty list
EMOTION_CLASSES_FOR_ANNOTATION = [
    "Neutral_Calm", # Key z
    "Alert_Curious",# Key x
    "Playful_Active",# Key c
    "Stressed_Agitated" # Key v
]
# ======================================================
SEQUENCE_LENGTH = 30

# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOG_LEVEL = "INFO"
TEST_VIDEO_FOLDER_NAME = "TestSet_Videos"

print("Config loaded (Custom CNN Detector mode).")
# ... (Your existing sanity checks at the end of config.py) ...
# (Ensure BBOX_FEATURE_LENGTH check is `if 'BBOX_FEATURE_LENGTH' not in globals(): ...`)
if not os.path.isdir(DATA_DIR) or (os.path.isdir(DATA_DIR) and not os.listdir(DATA_DIR)):
    print(f"WARNING: DATA_DIR ('{DATA_DIR}') is empty or does not exist. Please add your video files there.")
if not os.path.exists(DETECTION_MODEL_ONNX_VP):
    print(f"WARNING: Initial VP detection model not found: {DETECTION_MODEL_ONNX_VP}.")
if not os.path.exists(DETECTION_CLASSES_FILE_VP):
    print(f"WARNING: VP Detection classes file not found: {DETECTION_CLASSES_FILE_VP}.")
if 'BBOX_FEATURE_LENGTH' not in globals() and 'BBOX_FEATURE_LENGTH' not in locals(): # Corrected check
    print("CRITICAL WARNING: BBOX_FEATURE_LENGTH is not defined in this config module! This is required.")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env. GeminiHandler will not function.")
print(f"Effective DATA_DIR being used: {os.path.abspath(DATA_DIR)}")
print(f"Models directory: {os.path.abspath(MODELS_DIR)}")
print(f"CNN Patches will be stored under: {os.path.abspath(CNN_PATCHES_DIR)}")
print(f"Trained CNN model will be saved/loaded from: {os.path.abspath(CNN_MODEL_SAVE_PATH)}")
print(f"Behavior model (bbox-based) will be saved/loaded from: {os.path.abspath(BEHAVIOR_MODEL_BBOX_PATH)}")