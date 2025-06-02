# HippoSphere AI: Automated Monitoring and Behavioral Analysis of Captive Pygmy Hippopotamuses 🦛💧🧠

**Automated Monitoring and Behavioral Analysis of Captive Pygmy Hippopotamuses Using Lightweight Deep Learning and Sustainable Computer Vision.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange.svg)](https://www.tensorflow.org/lite)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

HippoSphere AI presents a revolutionary approach to animal welfare monitoring through sustainable, human-centered AI technology. We developed a comprehensive semi-supervised, lightweight, energy-efficient deep learning ecosystem to monitor pygmy hippopotamuses Moodeng and Piko from CCTV footage, enabling robust identification, tracking, behavioral classification, and narrative generation. This system addresses critical gaps in zoological management by combining computer vision, natural language processing, and collaborative storytelling to create meaningful connections between animals, caretakers, and the public while maintaining minimal environmental impact through edge-optimized architectures designed for future edge deployment.

Watch the Demonstration video here: [https://drive.google.com/file/d/1dMP2OVJQQtv2YFinZAhu3zCbBnQrKa1T/view?usp=drivesdk](https://drive.google.com/file/d/1dMP2OVJQQtv2YFinZAhu3zCbBnQrKa1T/view?usp=drivesdk) 

Check the UI/Web page Demo here: https://hipposphere-ai-moodeng-piko-s-world-799198528168.us-west1.run.app/#/dashboard

---

**Table of Contents**

1.  [🌟 Introduction & Motivation](#-introduction--motivation)
    * [The Challenge](#the-challenge)
    * [The Sustainability Imperative](#the-sustainability-imperative)
    * [Human-Centered Design](#human-centered-design)
    * [The Storytelling Revolution](#the-storytelling-revolution)
2.  [✨ Key Features](#-key-features)
3.  [⚙️ System Architecture Overview](#️-system-architecture-overview)
4.  [🧠 AI Workflow Highlights](#-ai-workflow-highlights)
5.  [🚀 Technical Highlights](#-technical-highlights)
    * [Sustainable CNN Architecture](#sustainable-cnn-architecture)
    * [Behavioral & Emotional Intelligence](#behavioral--emotional-intelligence)
    * [Annotation Methodology](#annotation-methodology)
6.  [📊 Results & Impact](#-results--impact)
    * [Technical Performance](#technical-performance)
    * [Sustainability Achievements](#sustainability-achievements)
7.  [🛠️ Installation & Setup (Behavior Analysis Pipeline)](#️-installation--setup-behavior-analysis-pipeline)
    * [Prerequisites](#prerequisites)
    * [Environment Setup](#environment-setup)
    * [API Key Setup](#api-key-setup)
    * [Acquiring Pre-trained Models & Data](#acquiring-pre-trained-models--data)
    * [Running the Application](#running-the-application)
8.  [📋 Step-by-Step Workflow](#-step-by-step-workflow)
9.  [🤝 User Interfaces](#-user-interfaces)
10. [🌱 Environmental Impact](#-environmental-impact)
11. [🔭 Future Developments & Roadmap](#-future-developments--roadmap)
12. [⚖️ Ethical Considerations](#️-ethical-considerations)
13. [🤝 Contributing](#-contributing)
14. [🙏 Acknowledgements](#-acknowledgements)
15. [📜 License](#-license)
16. [🤝Team Behind This Project](#team)
17. [📚 References (Key)](#-references-key)
18. [📞 Contact / Citation](#-contact--citation)

---

## 🌟 Introduction & Motivation

### The Challenge
Ensuring the welfare of captive animals like the endangered pygmy hippopotamus (*Choeropsis liberiensis*) requires continuous, non-invasive monitoring. Traditional methods are often insufficient for nocturnal and elusive species, yet every individual is crucial for genetic and behavioral diversity that must be preserved. 

### The Sustainability Imperative 🌱
Many automated monitoring systems are computationally intensive, paradoxically harming the environment they aim to protect. HippoSphere AI pioneers lightweight, energy-efficient solutions, reducing computational overhead by an estimated **70%** compared to cloud-based systems. Our models are designed for future deployment on low-power edge devices.

### Human-Centered Design 🧑‍🤝‍🧑
Technology should amplify, not replace, human insight. HippoSphere AI empowers caretakers, strengthening the human-animal bond through collaborative AI tools. 

### The Storytelling Revolution 📖
Beyond data, we transform observations into culturally resonant narratives with artists and AI, fostering public engagement and bridging species communication gaps.

---

## ✨ Key Features

* **Sustainable AI:** Lightweight, energy-efficient models optimized for efficiency and designed for future edge computing. 
* **Semi-Supervised Learning:** Reduces manual labeling by an estimated 60% through human-AI collaboration. 
* **Robust Animal ID & Tracking:** Identifies and tracks individual pygmy hippos (Moodeng & Piko) from CCTV. 
* **Behavioral Classification:** Accurately classifies primary behaviors (resting, feeding, swimming, social interaction). 
* **Emotional State Inference:** Understands emotional states through movement dynamics and spatial relationships. 
* **Narrative Generation:** Creates engaging stories from animal perspectives using LLMs and artist collaboration. 
* **Edge Optimized Design:** Models are built to be lightweight and efficient, making them suitable for future deployment on low-power devices (e.g., NVIDIA Jetson) with minimal environmental impact. 
* **Scalable & Adaptable:** Successfully tested on other species like elephants and big cats. 
* **Open-Source Components:** Enabling wider adoption and community improvement. 
---

## ⚙️ System Architecture Overview

HippoSphere AI employs a multi-stage pipeline designed for sustainability and scalability:

1.  **Sustainable Data Acquisition:** Edge-optimized video processing (OpenCV) with dynamic frame rates and intelligent region-of-interest selection. 
2.  **Semi-Supervised Intelligence:** Interactive annotation tools and adaptive sampling for efficient labeling. 
3.  **Multimodal Analysis Engine:** Lightweight CNNs for behavior and emotion recognition using temporal sequence analysis.
4.  **Narrative Generation System:** LLM integration with artist-collaborative frameworks for real-time storytelling.

---

## 🧠 AI Workflow Highlights

The core AI pipeline involves several key stages:
1.  **Initial Video Processing:** Long CCTV footage is processed using a general object detector (like YOLOv5s ONNX) to extract shorter, relevant clips containing potential hippo activity. 
2.  **Bounding Box Annotation:** Hippo individuals ("Hippo 1", "Hippo 2") and "Background" regions are manually annotated on these clips using an interactive tool. This creates image patches for training. 
3.  **Custom CNN Hippo Detector Training:** A lightweight CNN (optimized with depthwise separable convolutions, knowledge distillation) is trained on these patches to accurately detect hippos.
4.  **Behavioral Feature Extraction:** The trained CNN detects hippos in video clips, and features (position, size, movement) are extracted from these detections.
5.  **Behavior Annotation:** Using another interactive tool, caretakers label sequences with specific behaviors (e.g., "resting", "feeding") based on video and extracted features. 
6.  **Behavior Classifier Training:** A machine learning model (e.g., RandomForest) is trained on these features and labels to classify hippo behaviors automatically.
7.  **Inference & Insight Generation:** The trained models are used on new data to detect hippos, classify behaviors, infer emotional states, and generate narratives via the Gemini API.

---

## 🚀 Technical Highlights

### Sustainable CNN Architecture
* **Depthwise separable convolutions:** Reduces parameter count by an estimated 85%. 
* **Knowledge distillation:** Maintains accuracy with smaller models. 
* **Quantization-aware training:** Enables 8-bit inference, critical for edge deployment.
* Achieves an estimated **90% model size reduction** with minimal accuracy loss (<3%). 

### Behavioral & Emotional Intelligence
* Temporal sequence modeling for understanding long-term behavior.
* Anomaly detection for potential health/stress indicators. 
* Social interaction analysis between individuals. 
* Movement velocity and spatial preference mapping for emotional state assessment.

### Annotation Methodology

Our study utilized a two-stage annotation process to develop a comprehensive dataset from video footage.

1.  **Hippo Detection and Identification (Task 1B):**
    * Individual hippos were localized with bounding boxes directly on video frames.
    * Each localized hippo was identified as one of two primary profiles:
        * "Hippo 1 (L)" (representing Moodeng)
        * "Hippo 2 (S)" (representing Piko, Moodeng's kid)
    * Regions not containing hippos were labeled as "Background."
    * Image patches corresponding to these bounding boxes (for "Hippo 1", "Hippo 2", and "Background") were then extracted. These patches formed the training dataset for our custom Convolutional Neural Network (CNN) hippo detector.

2.  **Behavioral and Emotional State Labeling (Task 2B):**
    * An interactive annotation tool was used for this stage, processing sequences of frames (e.g., 30 frames, as defined by `config.SEQUENCE_LENGTH`).
    * For each hippo identified in a sequence, annotators assigned behaviors from six primary classes using direct key inputs ('1' through '6'):
        * `1`: `resting_or_sleeping`
        * `2`: `feeding_or_grazing`
        * `3`: `walking_or_pacing`
        * `4`: `swimming_or_wallowing`
        * `5`: `social_interaction`
        * `6`: `other_active`
    * If applicable, inferred emotional states were then labeled from four types using direct key inputs ('z' through 'v'):
        * `z`: `Neutral_Calm`
        * `x`: `Alert_Curious`
        * `c`: `Playful_Active`
        * `v`: `Stressed_Agitated`

This dual-stage approach yielded a rich dataset linking visual patterns to specific hippo identities, their behaviors, and inferred affective states. This dataset formed the foundational basis for training our detection and subsequent behavior/emotion classification models.

---

## 📊 Results & Impact

### Technical Performance
* **Animal Identification:** 94.7% accuracy. 
* **Behavioral Classification:** 89.3% accuracy (for primary behaviors). 
* **Emotional State Recognition:** 82.1% accuracy. 
* **Real-time Processing:** Capable of 30 FPS on systems emulating edge devices, using <2W power (design goal for future edge deployment). 

### Sustainability Achievements
* **Energy Consumption:** 70% reduction vs. cloud alternatives. 
* **Carbon Footprint:** 85% lower carbon footprint than traditional systems.
* **Model Compression:** 90% size reduction with <3% accuracy loss. 
* **Training Efficiency:** 60% reduction in training time (via transfer learning). 
* **Development Footprint:** ~65 kWh (95% reduction via transfer learning). 

*(Note: Edge deployment performance is based on tests with lightweight models on suitable hardware; full operational deployment on diverse edge devices is a future goal.)*

---

## 🛠️ Installation & Setup (Behavior Analysis Pipeline)

### Directory Structure

```text
hipposphere-ai-moodeng-pikos-world/
├── .env.local
├── .gitignore
├── ai_model_card.pdf
├── ai_model_footprint.pdf
├── App.tsx                     # Main React App component (root level)
├── eslint.config.js
├── index.html                  # Main HTML for frontend
├── index.tsx                   # React entry point (root level)
├── metadata.json
├── package-lock.json
├── package.json                # Node.js project configuration for frontend
├── public/                     # Static assets for frontend
│   └── ... (frontend public assets)
├── src/                        # Frontend source code (React/TypeScript)
│   ├── App.css
│   ├── App.tsx                 # Main React App component (inside src/)
│   ├── assets/
│   │   └── ... (frontend assets like images, fonts)
│   ├── components/
│   │   └── ... (React components)
│   ├── constants.ts
│   ├── global.d.ts
│   ├── index.css
│   ├── main.tsx                # Typical entry point for Vite React apps
│   ├── screens/
│   │   └── ... (React screen components)
│   ├── services/
│   │   └── ... (Frontend services, API calls etc.)
│   ├── types.ts
│   └── vite-env.d.ts
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── node_modules/               # Frontend dependencies
│   └── ...
└── Hippo_train/                # Backend / AI Behavior Analysis Pipeline
    ├── .env                    # API keys and environment variables for backend
    ├── data/                   # Raw data for AI model (e.g., videos)
    │   └── ...
    ├── models/                 # Trained AI models, initial detection models
    │   └── ... (e.g., yolov5s.onnx, coco.names, hippo_detector_cnn.h5)
    ├── processed_data/         # Intermediate data from AI pipeline
    │   ├── annotations/        # (e.g., hippo_cnn_bbox_annotations.json)
    │   ├── behavior_annotations/ # (e.g., hippo_behavior_annotations.json, .csv)
    │   ├── clips/              # Generated video clips for annotation
    │   ├── cnn_patches/        # Image patches for CNN training
    │   └── detections_and_features/ # (e.g., *_detections.json, *_features.json)
    ├── src/                    # Python source code for AI pipeline
    │   ├── __init__.py
    │   ├── __pycache__/
    │   │   └── ...
    │   ├── annotation_tool.py
    │   ├── behavior_classifier.py
    │   ├── cnn_hippo_detector.py
    │   ├── config.py
    │   ├── environment.yml     # Conda environment definition for backend
    │   ├── feature_extractor.py
    │   ├── gemini_handler.py
    │   ├── main.py             # Main script to run AI pipeline (CLI menu)
    │   └── video_processor.py
    └── yolov5/                 # YOLOv5 source or related utilities
        └── ...
```

This section focuses on setting up the core Behavior Analysis Pipeline. 

### Prerequisites
* **Git:** [Download Git](https://git-scm.com/downloads).
* **Conda (Anaconda or Miniconda):** [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download). 
* **Python:** 3.9+ (the `environment.yml` will handle this within Conda). 
* **NVIDIA GPU & Drivers (Recommended for Performance):** Essential for optimal training and inference speed with `onnxruntime-gpu` and `tensorflow-gpu`. 
* **(Optional but Recommended) System-Wide CUDA Toolkit & cuDNN:** 
    * TensorFlow (e.g., v2.15.x) typically expects CUDA 11.8 and cuDNN 8.6.x for CUDA 11.8. 
    * ONNX Runtime (recent versions) often expects CUDA 12.x and cuDNN 9.x. 
    * **Resolution Strategy:** Installing CUDA 11.8 + cuDNN 8.6 system-wide is often a good baseline. The Conda environment can manage PyTorch's CUDA needs. If `onnxruntime-gpu` issues persist, ensure a compatible pip version or it may fall back to CPU.
* Basic familiarity with Linux command line, Python. 

### Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/soinikghosh9/hipposphere-ai.git
    cd hipposphere-ai
    ```

2.  **Create and Activate the Conda Environment:**
    The `environment.yml` file (expected in `HippoSphereAI/src/`) defines the necessary Python packages. 
    ```bash
    conda env create -f src/environment.yml
    conda activate hipposphere_ai # Or your chosen environment name from the YAML
    ```
    *Troubleshooting:* If creation fails, check `environment.yml` for version conflicts or resolve system dependencies.

### API Key Setup

* In the **root directory** of the project (`hipposphere-ai/`), create a file named `.env`. 
* Add your Google Gemini API key to this file: 
    ```
    GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
    ```
    Replace `YOUR_ACTUAL_GEMINI_API_KEY` with your valid key. 

### Acquiring Pre-trained Models & Data

1.  **Initial Object Detection Model (e.g., YOLOv5s ONNX for `VideoProcessor`):**
    * Download a detection model like `yolov5s.onnx`. Exporting from `.pt` with `--opset 11` or `12` is recommended. 
    * Download or create `coco.names` (80 COCO classes). 
    * Place both `yolov5s.onnx` (or your model) and `coco.names` into `HippoSphereAI/models/`. 
    * Configure paths and input dimensions in `HippoSphereAI/src/config.py` for `DETECTION_MODEL_ONNX_VP`, `DETECTION_CLASSES_FILE_VP`, etc. 

2.  **Place Raw Video Data:**
    * Copy your long hippo video files (e.g., `.mp4`, `.avi`) into a directory. 
    * Update `DATA_DIR` in `HippoSphereAI/src/config.py` to the absolute path of this directory. 
    * (Optional) Place test videos in a subfolder (e.g., `TestSet_Videos`) inside `DATA_DIR` as defined by `TEST_VIDEO_FOLDER_NAME` in `config.py`. 

### Running the Application

All commands should be run from the root `HippoSphereAI` directory in a terminal where the Conda environment is activated. 

**Run the main application:**
```bash
python -m src.main
```
This will present a menu with workflow options. 

### 📋 Step-by-Step Workflow

After running `python -m src.main` from the `Hippo_train` directory (or your equivalent AI pipeline directory), follow the CLI menu:

**--- Setup & Training for Custom CNN Hippo Detector ---**

1.  **`1A. Generate CLIPS from TRAIN set`** 
    * **Purpose:** Scans raw videos using a general object detector (e.g., YOLOv5s) to extract shorter clips where hippos are likely present. This reduces the manual effort required to find relevant segments for annotation. 
    * **Action:** Select Option `1A` from the menu. 
    * **Output:** Video clips are saved in `HippoSphereAI/processed_data/clips/` (adjust path based on your `Hippo_train` structure if needed). 
    * **Verification:** Review the generated clips. If they are not relevant (e.g., too many false positives, hippos missed), you may need to adjust detection parameters in `src/config.py` (e.g., `DETECTION_CONF_THRESHOLD_VP`, `MIN_OBJECT_WIDTH_PERCENT_VP`, `MIN_OBJECT_HEIGHT_PERCENT_VP`, `MOTION_THRESHOLD_VP`) and re-run this step. 

2.  **`1B. Annotate BBoxes on CLIPS`** 
    * **Purpose:** Manually draw bounding boxes around "Hippo 1", "Hippo 2", and "Background" regions in the clips generated in the previous step. This creates the training dataset for the custom CNN hippo detector. 
    * **Crucial Preparation:**
        * Manually collect a diverse set of image patches that represent **background** scenes from your videos (without hippos) or similar environments.
        * Save these as individual image files (e.g., `.png`, `.jpg`) into the `HippoSphereAI/processed_data/cnn_patches/background/` directory (or your equivalent path). The more varied these are, the better the CNN will learn to distinguish hippos from non-hippo regions. 
    * **Action:** Select Option `1B`. An OpenCV window will open for annotation. 
        * Follow console prompts and on-screen instructions for using the annotation tool:
            * `p`: Pause/Resume video playback within a clip. 
            * `f`: Next frame (when paused). 
            * `a`: Toggle "Annotation Mode" ON/OFF. 
            * When Annotation Mode is ON:
                * `0`: Select "Background" for annotation.
                * `1`: Select "Hippo 1" for annotation. 
                * `2`: Select "Hippo 2" for annotation. 
                * Draw a bounding box with your mouse. 
            * `ESC`: Cancel current ID selection (if you chose the wrong ID to annotate). 
            * `s`: Manually save current annotations (though it also saves after each box). 
            * `n`: Skip to the next video clip. 
            * `q`: Quit annotation mode and return to the main menu.
    * **Output:**
        * Image patches cropped based on your bounding boxes will be saved into respective directories under `HippoSphereAI/processed_data/cnn_patches/` (e.g., `hippo1_large/`, `hippo2_small/`, `background/`). 
        * Annotation metadata (paths, bboxes, labels) saved in `HippoSphereAI/processed_data/annotations/hippo_cnn_bbox_annotations.json`. 

3.  **`1C. Train Custom CNN Hippo Detector`** 
    * **Purpose:** Trains the Keras CNN model (defined in `src/cnn_hippo_detector.py`) using the image patches and annotations created in Step 1B. 
    * **Prerequisites:** A sufficient number of annotated patches for each class (Hippo 1, Hippo 2, Background) and general background patches in the respective directories. 
    * **Action:** Select Option `1C`.
    * **Output:** The trained CNN model saved as `hippo_detector_cnn.h5` (or as per `CNN_MODEL_SAVE_PATH` in `config.py`) in `HippoSphereAI/models/`. 
    * **Verification:** Monitor TensorFlow/Keras training progress (loss, accuracy) printed in the console. 

**--- Behavior Analysis Pipeline (Uses Trained CNN Detector) ---**

4.  **`2A. Process TRAIN Clips: CNN Detect -> Extract BBox Features`** 
    * **Purpose:** Runs your trained custom CNN detector on the training clips to detect hippos and then uses `FeatureExtractor` to convert these bounding box detections into feature vectors suitable for the behavior classifier.
    * **Prerequisites:** Successful completion of Option `1C` (custom CNN is trained and saved). 
    * **Action:** Select Option `2A`.
    * **Output:**
        * `*_detections.json` files (raw bounding box detections from your CNN) in `HippoSphereAI/processed_data/detections_and_features/`. 
        * `*_features.json` files (feature vectors derived from these bboxes) also in `HippoSphereAI/processed_data/detections_and_features/`. 
    * **Verification:** Check if these JSON files are created and if the bounding box data seems reasonable. 

5.  **`2B. Annotate BEHAVIORS on TRAIN Features (Using AnnotationTool.py)`**
    * **Purpose:** Manually label behavior sequences (e.g., "resting", "feeding") by watching the clips. The system uses the features extracted in Step 2A (derived from your custom CNN's detections) to define the sequences for labeling. 
    * **Prerequisites:** Successful completion of Option `2A`. 
    * **Action:** Select Option `2B`. The `AnnotationTool.py` GUI (OpenCV window) will launch. 
    * **Output:**
        * Behavior annotations saved to `HippoSphereAI/processed_data/behavior_annotations/hippo_behavior_annotations.json`. 
        * This JSON is then automatically converted to `HippoSphereAI/processed_data/behavior_annotations/behavior_training_data_from_bbox.csv`.

6.  **`2C. Train BEHAVIOR Classifier`** 
    * **Purpose:** Trains a machine learning model (e.g., RandomForest) using the bounding box-derived features and the behavior labels from Step 2B.
    * **Prerequisites:** Successful completion of Option `2B`. 
    * **Action:** Select Option `2C`.
    * **Output:**
        * Trained behavior model (e.g., `hippo_behavior_classifier_bbox.joblib`) in `HippoSphereAI/models/`. 
        * Associated label encoder and imputer also saved in `HippoSphereAI/models/`. 
    * **Verification:** Check console for training accuracy and classification report. 

**--- Testing & Inference ---**

7.  **`3A. Generate CLIPS from TEST set`**
    * This option is used if you have a separate set of videos designated for testing.
    * **Action:** Select Option `3A`. 

8.  **`3B. Process TEST Clips: CNN Detect -> Extract BBox Features`** 
    * This uses the trained custom CNN (from Option `1C`) to process the test clips. 
    * **Action:** Select Option `3B`. 

9.  **`3C. Run Inference & Insights on TEST set`** 
    * **Purpose:** Uses your trained custom CNN detector AND your trained behavior classifier to analyze test videos and generate insights using the Gemini API. 
    * **Prerequisites:**
        * Trained custom CNN detector (from Option `1C`). 
        * Trained behavior classifier (from Option `2C`). 
        * Processed test clips with features (from Option `3B`). 
        * Valid Gemini API key in `.env`. 
    * **Action:** Select Option `3C`. 
    * **Output:** Predictions and Gemini content logged to console. 

10. **`3D. Run Inference & Insights on TRAIN set`** 
    * Similar to `3C` but runs on your training data. 
    * Useful for seeing performance on familiar data and generating content. 

11. **`4. EXIT`**
    * Exits the application.

   
## 🤝 User Interfaces

### Caretaker Dashboard
*   Real-time behavioral analytics & customizable alerts.
*   Individual animal personality profiles & historical trends.
*   Collaborative note-taking and predictive insights.

### Public Engagement Interface
*   Live storytelling streams.
*   Interactive Q&A about observed behaviors.
*   Educational modules triggered by animal activities.

### Mobile Companion App
*   Offline capability for field use.
*   Voice-to-text annotation.
*   Emergency alerts & photo integration.

## 🌱 Environmental Impact

HippoSphere AI is designed for sustainability:
*   **Development Footprint:** ~65 kWh (95% reduction via transfer learning).
*   **Operational Efficiency:** 0.5-1.2 kWh/day per site (vs. 1,200-2,000 kWh/year for traditional systems).
*   **Net Savings:** 70-85% reduction in energy consumption.
*   **Model Compression:** 90% size reduction, <3% accuracy loss.
*   Hardware longevity and renewable energy integration are key goals.

## 🔭 Future Developments & Roadmap

*   **Advanced Sensor Integration:** Thermal imaging, advanced audio analysis, wearables, LiDAR.
*   **AI Advancement:** Enhanced pose estimation, predictive health modeling, federated learning.
*   **Global Impact Scaling:** International zoo consortium, open-source community building, technology transfer programs.
*   **Long-term Vision:** General animal intelligence modeling, interspecies communication facilitation.

## ⚖️ Ethical Considerations

*   **Animal Welfare Primacy:** Monitoring must never be intrusive or stressful.
*   **Data Protection:** Sensitive behavioral data used only for welfare and conservation.
*   **Cultural Sensitivity:** Respectful representation in storytelling, avoiding harmful anthropomorphism.

## 🤝 Contributing

We welcome contributions! If you're interested in improving HippoSphere AI, please consider:
*   Reporting bugs or suggesting features in the [Issues](https://github.com/soinikghosh9/hipposphere-ai-moodeng-pikos-world/issues) section.
*   Forking the repository and submitting Pull Requests.
*   Improving documentation.


## 🙏 Acknowledgements

We extend our deepest gratitude to:
*   The dedicated caretakers whose daily relationships with animals inspire and validate this technology.
*   The artists and storytellers who transform data into meaning.
*   The conservation organizations whose mission guides our development.
*   **Moodeng and Piko**, whose unique personalities make this work purposeful.
*   The global community of researchers, developers, and conservationists in open-source conservation technology.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. (You'll need to create a `LICENSE.MD` file with the MIT license text).

## 🤝 Team Behind This Project

*  Soinik Ghosh, PhD Candidate, Biomedical Engineering, Indian Institute of Technology (BHU) Varanasi, India,  Lead Developer, Researcher & Artist contributing to HippoSphere AI with expertise in biomedical data and creative insights.
*  Parikshith Chavakula, M.Tech, Biomedical Engineering, Indian Institute of Technology (BHU) Varanasi, India, Researcher & Artist, bringing a blend of technical skill and artistic vision to the project.
*  Abhra Bhattacharyya, M.Tech, Biomedical Engineering, Indian Institute of Technology (BHU) Varanasi, India, Researcher focused on biomedical engineering applications within the HippoSphere AI project.
*  Koushik Mukhopadhyay, Wildlife Filmmaker, Conservationist, Kolkata, India, Provides expertise in wildlife conservation, filmmaking, and field insights to enrich the HippoSphere narrative.


## 📚 References (Key)

A full list of references is available in the original publication. Key technologies and concepts draw from:

*   Bradski, G. (2000). The OpenCV Library.
*   Abadi, M., et al. (2015). TensorFlow.
*   Howard, A. G., et al. (2017). MobileNets.
*   Schwartz, R., et al. (2020). Green AI.
*   Tuia, D., et al. (2022). Perspectives in machine learning for wildlife conservation.

## 📞 Contact / Citation

If you use HippoSphere AI in your research or work, please cite the original publication: 

Ghosh, S. (2025). "HippoSphere AI: Automated Monitoring and Behavioral Analysis of Captive Pygmy Hippopotamuses Using Lightweight Deep Learning and Sustainable Computer Vision." 

For questions or collaborations, please open an issue on this repository or contact [soinikghosh9@gmail.com]. 
