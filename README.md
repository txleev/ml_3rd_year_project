📄 Final Project Report: Deep Learning – Emotion Detection from Video Using YOLO
1. Project Proposal 
Title:
Real-Time Emotion Detection in Video Using YOLOv8 and Facial Expression Recognition

Background & Problem Statement:
Understanding human emotions from facial expressions has wide applications in mental health monitoring, HCI, security, and entertainment. Manual analysis of emotions in videos is time-consuming and subjective.

Objective:
To design and train a YOLOv8-based object detection model that detects faces and classifies their emotion in video frames, enabling sentiment analysis at scale.

Expected Outcomes:

A fine-tuned YOLOv8 model capable of detecting facial emotions in real-time.

A full pipeline to run inference on video and extract per-class emotion counts.

Insights about the general sentiment expressed in the video.

2. Data Collection & Preprocessing 
Dataset:
We used the [AffectNet] dataset, which contains labeled facial images across multiple emotion classes like happy, sad, angry, surprised, neutral, etc.

Preprocessing Steps:

Converted dataset into YOLO format with .txt annotations.

Split dataset into train/val/test.

Resized all images to 640x640 resolution.

Applied normalization, face alignment (optional), and data augmentation (flip, brightness).

Documentation:
All data transformations and file structure were maintained in a YOLO_format/ directory with a proper data.yaml config file.

3. Exploratory Data Analysis 
Performed:

Class distribution analysis (some emotions were underrepresented).

Sample image visualization to confirm correct bounding boxes and labels.

Visual verification of emotion classes (e.g., confusion between "neutral" and "sad").

Insights:

Dataset was slightly imbalanced, which we considered during training.

Augmentation helped mitigate the skew.

4. Model Selection & Justification 
Model Chosen: YOLOv8 (Nano variant for real-time inference)

Justification:

YOLOv8 offers excellent speed–accuracy tradeoff.

Ideal for object detection + classification in videos.

Custom training is straightforward, and export to ONNX/TFLite is available for deployment.

5. Model Training & Validation
Training Details:


model = YOLO("yolov8n.pt")
results = model.train(
    data="YOLO_format/data.yaml",
    epochs=25,
    imgsz=640,
    batch=32,
    half=True,
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    dropout=0.0,
    name="affectnet_rtx3070"
)
Validation:
The model was validated using mAP, precision, recall. Sample results:

mAP@0.5: 0.72

Precision: 0.68

Recall: 0.71

6. Hyperparameter Tuning
We tuned the following:

Batch size (limited by GPU memory — 32 was optimal)

Epochs: Extended to 25 after observing stabilization

Dropout: Tried 0.1, but 0.0 performed better

Warmup epochs helped avoid early instability

Improvements:

mAP improved by ~5% after tuning learning rate and warmup settings.

7. Deployment Demonstration 
A complete pipeline was developed to:

Load a video

Process each frame through YOLO

Display results with emotion predictions

Count each detected emotion and store in a dictionary

Sample output:

Detected object counts: {'happy': 112, 'sad': 65, 'neutral': 201}
From this we inferred:

The dominant emotion in the video was "neutral", followed by "happy".

8. Documentation & Code Quality 
✅ All code was modular, clean, and well-commented.
✅ Jupyter notebook included with:

Image prediction

Video prediction

Visualization of detections
✅ Inference results were visualized inline using matplotlib.
✅ Dependencies were listed in requirements.txt.

9. Final Presentation & Report 
This report, along with the final presentation, includes:

Problem motivation

Training methodology

Sample predictions

Inference demonstration

Insights into emotion trends in videos

