# src/behavior_classifier.py
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import logging
from sklearn.impute import SimpleImputer

from src import config # Uses config.BBOX_FEATURE_LENGTH, config.SEQUENCE_LENGTH, 
                       # config.BEHAVIOR_MODEL_BBOX_PATH, etc.

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class BehaviorClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.imputer = None
        self._load_model()

    def _load_model(self):
        # === UPDATED to use new config paths for BBox-based model ===
        model_path = config.BEHAVIOR_MODEL_BBOX_PATH
        encoder_path = config.BEHAVIOR_LABEL_ENCODER_BBOX_PATH
        imputer_path = config.BEHAVIOR_IMPUTER_BBOX_PATH # Use new config var
        # ============================================================
        
        models_exist = True
        if not os.path.exists(model_path):
            logger.info(f"Behavior classification model not found at: {model_path}")
            models_exist = False
        if not os.path.exists(encoder_path):
            logger.info(f"Behavior label encoder not found at: {encoder_path}")
            models_exist = False
        
        if models_exist:
            try:
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(encoder_path)
                if os.path.exists(imputer_path): # Imputer is optional but recommended
                    self.imputer = joblib.load(imputer_path)
                    logger.info(f"Behavior imputer loaded from {imputer_path}")
                else:
                    logger.warning(f"Behavior imputer not found at {imputer_path}. Will create new one if training.")
                
                logger.info(f"Behavior classification model loaded from {model_path}")
                logger.info(f"Label encoder loaded from {encoder_path}")
            except Exception as e:
                logger.error(f"Error loading behavior model/encoder/imputer: {e}")
                self.model = None; self.label_encoder = None; self.imputer = None
        else:
            logger.info("Behavior model and/or label encoder not found. Model needs to be trained.")


    def _preprocess_features_for_training(self, features_str_list):
        """Converts a list of stringified feature vectors into a list of NumPy arrays."""
        processed_features = []
        # === UPDATED EXPECTED LENGTH CALCULATION using config.BBOX_FEATURE_LENGTH ===
        if not hasattr(config, 'BBOX_FEATURE_LENGTH'):
            logger.critical("CRITICAL: config.BBOX_FEATURE_LENGTH is not defined! Cannot determine feature length.")
            # Or raise an error: raise AttributeError("config.BBOX_FEATURE_LENGTH not defined!")
            return [] # Return empty, training will fail
        
        single_frame_feature_len = config.BBOX_FEATURE_LENGTH
        expected_len_one_sequence = single_frame_feature_len * config.SEQUENCE_LENGTH
        # ============================================================================
        
        for f_str in features_str_list:
            try:
                # Replace 'nan' string with np.nan for eval to work correctly with numpy
                # Also handle potential 'None' strings if they sneak in, though less likely
                f_str_cleaned = f_str.replace("'nan'", "np.nan").replace("nan", "np.nan").replace("None", "np.nan")
                f_list = eval(f_str_cleaned)
                f_array = np.array(f_list, dtype=float)

                if f_array.shape[0] != expected_len_one_sequence:
                    logger.warning(f"Feature vector sequence length mismatch after eval. "
                                   f"Expected {expected_len_one_sequence}, got {f_array.shape[0]}. "
                                   f"Original string (first 100 chars): '{f_str[:100]}...'. Skipping.")
                    processed_features.append(np.full(expected_len_one_sequence, np.nan))
                else:
                    processed_features.append(f_array)
            except SyntaxError:
                logger.error(f"SyntaxError parsing feature string (likely malformed list): '{f_str[:100]}...'. Skipping.")
                processed_features.append(np.full(expected_len_one_sequence, np.nan))
            except Exception as e: 
                logger.error(f"General error parsing feature string '{f_str[:100]}...': {e}. Skipping.")
                processed_features.append(np.full(expected_len_one_sequence, np.nan))
        return processed_features

    def train(self, annotated_features_csv_path):
        # ... (Core training logic remains the same as your previous version) ...
        # ... (Loading df, splitting X_sequences_array, y_labels_str, encoding, imputation, train_test_split, model.fit, evaluation) ...
        # The key change was in _preprocess_features_for_training using BBOX_FEATURE_LENGTH

        logger.info(f"Starting behavior classifier training with data from: {annotated_features_csv_path}")
        if not os.path.exists(annotated_features_csv_path):
            logger.error(f"Annotation CSV file not found: {annotated_features_csv_path}"); return
        try:
            df = pd.read_csv(annotated_features_csv_path)
        except Exception as e:
            logger.error(f"Error reading annotation CSV file {annotated_features_csv_path}: {e}"); return
        if 'features' not in df.columns or 'label' not in df.columns:
            logger.error("Annotation CSV must contain 'features' and 'label' columns."); return
        df = df.dropna(subset=['label', 'features'])
        if df.empty: logger.error("No data after dropping NaNs. Cannot train."); return

        X_sequences_raw = self._preprocess_features_for_training(df['features'].tolist())
        if not X_sequences_raw: logger.error("Preprocessing features resulted in no data. Cannot train."); return
        X_sequences_array = np.array(X_sequences_raw)
        y_labels_str = df['label'].tolist()

        if X_sequences_array.ndim != 2 or X_sequences_array.shape[0] == 0 or X_sequences_array.shape[0] != len(y_labels_str):
            logger.error(f"Problem with X_sequences_array shape ({X_sequences_array.shape}) or y_labels_str length ({len(y_labels_str)}). Cannot train."); return
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_labels_str)
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_imputed = self.imputer.fit_transform(X_sequences_array)
        
        min_samples_per_class_for_stratify = 2 
        unique_labels, counts = np.unique(y_encoded, return_counts=True)
        stratify_param = y_encoded if all(c >= min_samples_per_class_for_stratify for c in counts) and len(unique_labels) > 1 else None
        if stratify_param is None: logger.warning("Using non-stratified split for train/test.")
        if len(unique_labels) <=1 and X_imputed.shape[0] > 0 :
             logger.error("Only one unique class or no classes found in labels after processing. Cannot train a meaningful classifier."); return

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.25, random_state=42, stratify=stratify_param)
        if X_train.shape[0] == 0: logger.error("No training samples after split."); return
        logger.info(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
        logger.info(f"Test data shape: X_test {X_test.shape}, y_test {y_test.shape}")
        
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=20, min_samples_split=5, min_samples_leaf=2, n_jobs=-1)
        self.model.fit(X_train, y_train)

        if X_test.shape[0] > 0:
            y_pred_test = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_test)
            logger.info(f"Model Accuracy on test set: {accuracy:.4f}")
            class_names_report = self.label_encoder.classes_
            report = classification_report(y_test, y_pred_test, target_names=class_names_report, zero_division=0)
            logger.info("Classification Report on Test Set:\n" + report)
        else:
            logger.warning("Test set is empty, skipping test set evaluation.")

        # === UPDATED Saving part to use new config paths ===
        model_save_path = config.BEHAVIOR_MODEL_BBOX_PATH
        encoder_save_path = config.BEHAVIOR_LABEL_ENCODER_BBOX_PATH
        imputer_save_path = config.BEHAVIOR_IMPUTER_BBOX_PATH # Use new config var
        # ==================================================

        joblib.dump(self.model, model_save_path)
        joblib.dump(self.label_encoder, encoder_save_path)
        joblib.dump(self.imputer, imputer_save_path) # Save the fitted imputer
        logger.info(f"Behavior model (bbox-based) saved to {model_save_path}")
        logger.info(f"Label encoder (bbox-based) saved to {encoder_save_path}")
        logger.info(f"Imputer (bbox-based) saved to {imputer_save_path}")

    def predict(self, feature_sequence_list_np):
        # ... (This method remains largely the same as your previous version, using self.imputer) ...
        # The key is that self.imputer is now loaded/saved consistently with the bbox model.
        if self.model is None or self.label_encoder is None:
            logger.error("Model or label encoder not loaded/trained. Cannot predict.")
            return ["model_not_loaded"] * len(feature_sequence_list_np) if isinstance(feature_sequence_list_np, (list,np.ndarray)) else ["model_not_loaded"]
        
        feature_array = np.array(feature_sequence_list_np, dtype=float) if not isinstance(feature_sequence_list_np, np.ndarray) else feature_sequence_list_np
        if feature_array.ndim == 1: feature_array = feature_array.reshape(1, -1)
        elif feature_array.size == 0: return []
        
        if self.imputer is None:
            logger.critical("CRITICAL: Imputer not available for prediction! This should have been loaded with the model. Results may be unreliable.")
            # Fallback: Create and fit a new one on the prediction data (highly not recommended for consistency)
            temp_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            try: feature_array_imputed = temp_imputer.fit_transform(feature_array)
            except ValueError: return ["imputation_error"] * feature_array.shape[0]
        else:
            try: feature_array_imputed = self.imputer.transform(feature_array)
            except Exception as e:
                logger.error(f"Error using stored imputer.transform: {e}. Trying fit_transform as fallback.")
                try: feature_array_imputed = self.imputer.fit_transform(feature_array)
                except ValueError: return ["imputation_error"] * feature_array.shape[0]
        try:
            predictions_encoded = self.model.predict(feature_array_imputed)
            predictions_labels = self.label_encoder.inverse_transform(predictions_encoded)
            return predictions_labels.tolist()
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return ["prediction_error"] * feature_array.shape[0]


if __name__ == '__main__':
    if not os.path.exists(".env"):
        with open(".env", "w") as f: f.write("GEMINI_API_KEY=YOUR_API_KEY_HERE\n")

    if not hasattr(config, 'BBOX_FEATURE_LENGTH'):
        print("ERROR for __main__ test: config.BBOX_FEATURE_LENGTH not defined.")
        exit()
    if not hasattr(config, 'HIPPO_PROFILES_CNN'): # Needed for hippo_id in dummy data
        print("ERROR for __main__ test: config.HIPPO_PROFILES_CNN not defined.")
        exit()

    # Use the BBOX-specific paths from config
    dummy_csv_annotations_file = os.path.join(config.BEHAVIOR_ANNOTATIONS_DIR, "dummy_bbox_behavior_training_data.csv")
    os.makedirs(config.BEHAVIOR_ANNOTATIONS_DIR, exist_ok=True)

    # Check against BBOX specific model paths
    if not (os.path.exists(config.BEHAVIOR_MODEL_BBOX_PATH) and \
            os.path.exists(config.BEHAVIOR_LABEL_ENCODER_BBOX_PATH)) or \
       not os.path.exists(dummy_csv_annotations_file):
        logger.info(f"Creating dummy training CSV data (bbox-feature based) at: {dummy_csv_annotations_file}")
        num_samples = 300
        flat_sequence_len = config.BBOX_FEATURE_LENGTH * config.SEQUENCE_LENGTH
        
        num_hippo_profiles = len(config.HIPPO_PROFILES_CNN) if config.HIPPO_PROFILES_CNN else 2 # Default to 2 if not set
        
        dummy_data_for_df = {
            'video_path': [f"vid_{i//60}.mp4" for i in range(num_samples)],
            'frame_idx': [i * config.SEQUENCE_LENGTH for i in range(num_samples)],
            'timestamp_sec': [i * config.SEQUENCE_LENGTH * (1.0/(config.FPS_FOR_POSE_ESTIMATION if hasattr(config, 'FPS_FOR_POSE_ESTIMATION') and config.FPS_FOR_POSE_ESTIMATION > 0 else 25.0)) for i in range(num_samples)],
            'hippo_id': [(i//30) % num_hippo_profiles for i in range(num_samples)],
            'features': [str(np.random.rand(flat_sequence_len).tolist()) for _ in range(num_samples)],
            'label': np.random.choice(config.BEHAVIOR_CLASSES[:-1], num_samples) # Exclude 'out_of_frame'
        }
        # ... (rest of your dummy data generation with NaNs, ensure eval string is correct) ...
        for i in range(num_samples // 10):
            idx_to_modify = np.random.randint(num_samples)
            list_features_str = dummy_data_for_df['features'][idx_to_modify]
            try:
                # Convert string to list, insert np.nan, then convert list back to desired string format
                list_features = eval(list_features_str) # String to list of floats
                if list_features and len(list_features) > 0:
                    nan_indices = np.random.choice(len(list_features), size=max(1, len(list_features)//20), replace=False)
                    for nan_idx in nan_indices:
                        list_features[nan_idx] = np.nan # Actual NaN for processing
                    # For saving to CSV, eval expects 'np.nan' or a convertible value, not just 'nan'
                    # The _preprocess_features_for_training handles "str(...).replace('nan', 'np.nan')"
                    # So, string representation should be like "[..., nan, ...]"
                    dummy_data_for_df['features'][idx_to_modify] = str(list_features) 
            except: pass

        dummy_df = pd.DataFrame(dummy_data_for_df)
        dummy_df.to_csv(dummy_csv_annotations_file, index=False)
        logger.info(f"Dummy training CSV created: {dummy_csv_annotations_file}")
        
        bc_train = BehaviorClassifier()
        bc_train.train(dummy_csv_annotations_file)
    else:
        logger.info("Behavior model (bbox-based) and/or dummy CSV already exist. Skipping dummy data creation/training.")

    bc_predict = BehaviorClassifier() # This will load the bbox-based model due to config changes
    if bc_predict.model and bc_predict.label_encoder:
        num_predict_sequences = 5
        sequence_flat_len = config.BBOX_FEATURE_LENGTH * config.SEQUENCE_LENGTH
        dummy_predict_data_np = [np.random.rand(sequence_flat_len) for _ in range(num_predict_sequences)]
        if dummy_predict_data_np and dummy_predict_data_np[0] is not None and sequence_flat_len > 0:
             dummy_predict_data_np[0][0:max(1,sequence_flat_len//10)] = np.nan

        predictions = bc_predict.predict(dummy_predict_data_np)
        logger.info(f"Dummy Predictions (bbox-feature based): {predictions}")
    else:
        logger.warning("Cannot run prediction example: model (bbox-feature based) not available/loaded.")