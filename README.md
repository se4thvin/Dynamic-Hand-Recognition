# Dynamic Hand Recognition for Solfege Detection

This project implements a hand gesture recognition system for detecting solfege syllables (`do`, `re`, `mi`, `fa`, `sol`, `la`, `ti`, `do`). It uses a deep learning approach with MediaPipe for hand landmark detection and TensorFlow for gesture classification. By leveraging hand landmarks rather than raw pixel data, this system achieves efficient and accurate gesture recognition with minimal training data.

---

## Features
- **Top-Level Hand Gesture Recognition**: Detects solfege hand gestures in real-time.
- **MediaPipe Integration**: Utilizes MediaPipe's hand detection and landmarking for efficient preprocessing.
- **Neural Network Classifier**: Classifies gestures into solfege syllables using a custom TensorFlow model.
- **Lightweight Training**: Reduces the need for large datasets by extracting hand landmarks instead of raw pixel matrices.

---

## How It Works

### Deep Learning Approach
1. **Hand Detection and Landmarking**:
   - MediaPipe detects hands and provides 21 hand landmarks.
   - Landmarks include key points like the wrist, finger joints, and fingertips.

2. **Preprocessing**:
   - **Relative Coordinates**:
     - The wrist (base point) is set as the origin `(0, 0)`, and all landmarks are shifted relative to this point.
     - Removes absolute location dependency.
   - **Normalization**:
     - Coordinates are flattened into a one-dimensional list for neural network input.
     - Values are scaled to the range `[-1, 1]` by dividing by the maximum absolute value of the coordinates.
   - This preprocessing reduces the complexity of the input data, making the system robust to variations in hand size, position, and orientation.

3. **Gesture Classification**:
   - A trained neural network processes normalized landmark data and classifies it into one of the solfege syllables.
   - The model is designed to learn a mapping between hand landmarks and solfege syllables, enabling accurate recognition.

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
This directory stores files related to finger gesture recognition.<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)
