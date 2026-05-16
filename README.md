# Traffic Sign Recognition using Deep Learning + Explainable AI

An end-to-end Computer Vision project for multiclass traffic sign classification using Convolutional Neural Networks (CNNs), deployed with Streamlit and enhanced with Grad-CAM Explainable AI visualizations.

This project goes beyond standard image classification by integrating:
- real-world deployment practices,
- explainability (XAI),
- confidence analysis,
- and engineering-level debugging/serialization handling for TensorFlow/Keras production environments.

---

# 🚀 Live Features

✅ Real-time traffic sign prediction  
✅ 43-class multiclass classification  
✅ Confidence score visualization  
✅ Top-3 prediction probabilities  
✅ Grad-CAM explainability heatmaps  
✅ Streamlit web deployment  
✅ Production-style preprocessing pipeline  
✅ TensorFlow/Keras deployment compatibility handling  

---

# 🧠 Project Motivation

Traffic sign recognition is a critical component of:
- Autonomous Driving
- ADAS (Advanced Driver Assistance Systems)
- Smart Transportation Systems
- Edge AI applications

The goal of this project was not only to build a high-performing CNN classifier, but also to simulate a real-world ML deployment workflow including:
- model serialization,
- deployment debugging,
- explainability integration,
- and inference pipeline engineering.

---

# 🖼️ Demo

## Prediction Interface

_Add screenshot here_

```markdown
![images/Predictions.png](https://github.com/IBOSS24/Traffic_sign_classification/blob/main/Images/Predictions.png)
```

---

## Confidence Visualization

_Add screenshot here_

```markdown
![Confidence Bars](images/demo2.png)
```

---

## Grad-CAM Explainability

_Add screenshot here_

```markdown
![GradCAM](images/gradcam_demo.png)
```

---

# 🏗️ Model Architecture

The model is a custom CNN architecture designed for traffic sign classification.

## Architecture Overview

```python
Conv2D(64) + ReLU
BatchNormalization
MaxPooling2D
Dropout(0.25)

Conv2D(64) + ReLU
BatchNormalization
MaxPooling2D
Dropout(0.25)

Flatten

Dense(128) + ReLU
Dropout(0.5)

Dense(43) + Softmax
```

---

# 🧪 Training Techniques Used

## Optimization
- AdamW optimizer
- Learning rate scheduling
- Weight decay regularization

## Regularization
- Dropout
- BatchNormalization

## Image Preprocessing
- Image resizing to `(50, 50)`
- Pixel normalization `[0,255] → [0,1]`

## Loss Function
- Sparse Categorical Crossentropy

---

# 📂 Dataset

The project uses the German Traffic Sign Recognition Benchmark (GTSRB), containing 43 traffic sign classes.

Typical classes include:
- Speed Limits
- Stop Signs
- Yield Signs
- No Entry
- Dangerous Curves
- Pedestrian Crossing

---

# 🧠 Explainable AI (XAI)

This project integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize:
- which image regions influenced the model prediction,
- and how the CNN focuses attention spatially.

This improves:
- model transparency,
- interpretability,
- and trustworthiness.

---

# ⚙️ Engineering Challenges Solved

One of the strongest aspects of this project was solving real-world deployment and serialization issues encountered during TensorFlow/Keras productionization.

## Key Challenges Encountered

### 1. TensorFlow/Keras Serialization Compatibility
While deploying the model locally, multiple incompatibilities appeared between:
- TensorFlow 2.15
- TensorFlow 2.16
- Keras 3.x
- Colab serialization formats

Errors included:
- `quantization_config`
- `batch_shape`
- `optional`
- model deserialization failures

### 2. Cross-Environment Model Portability
The model was originally trained in Google Colab but deployed locally using Streamlit.

To solve serialization instability:
- architecture recreation + weights-only deployment was implemented,
- bypassing fragile `.keras` and `.h5` deserialization workflows.

### 3. Keras 3 Grad-CAM Compatibility
Modern Keras versions changed symbolic graph handling, causing:
- `model.output` failures,
- eager vs symbolic tensor conflicts,
- disconnected gradient graphs.

The Grad-CAM pipeline was manually adapted for Keras 3 compatibility.

### 4. Production-Oriented Debugging
The project required debugging:
- model graph construction,
- gradient tracing,
- TensorFlow eager execution,
- layer connectivity,
- and deployment environment consistency.

These issues reflect real-world ML engineering workflows rather than tutorial-only development.

---

# 📊 Prediction Pipeline

## Workflow

```text
User Upload Image
        ↓
Image Preprocessing
        ↓
CNN Prediction
        ↓
Confidence Analysis
        ↓
Top-3 Predictions
        ↓
Grad-CAM Visualization
        ↓
Streamlit UI Output
```

---

# 🛠️ Tech Stack

## Deep Learning
- TensorFlow
- Keras

## Deployment
- Streamlit

## Computer Vision
- OpenCV
- Pillow

## Scientific Computing
- NumPy
- Matplotlib

---

# 📁 Project Structure

```text
traffic-sign-recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── traffic.weights.h5
│
├── images/
│   ├── demo1.png
│   ├── demo2.png
│   └── gradcam_demo.png
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── gradcam.py
│   ├── model_architecture.py
│   └── utils.py
│
└── data/
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone <your-repo-link>

cd traffic-sign-recognition
```

---

## Create Environment

```bash
conda create -n traffic_env python=3.11

conda activate traffic_env
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📈 Future Improvements

## Model Improvements
- EfficientNet / MobileNet transfer learning
- Data augmentation pipeline
- Hyperparameter optimization

## Deployment Improvements
- Docker containerization
- ONNX/TensorRT optimization
- GPU inference support

## Application Improvements
- Real-time webcam detection
- Video stream inference
- Mobile deployment

---

# 🧠 Lessons Learned

This project reinforced several critical Machine Learning Engineering concepts:

- Training environment ≠ deployment environment
- Serialization stability matters in production
- Explainability requires graph-level understanding
- Keras/TensorFlow versioning can significantly affect deployment
- Real-world ML involves debugging infrastructure as much as modeling

---

# 👨‍💻 Author

Mohammed Badr

Aspiring Machine Learning Engineer & Data Scientist  
Background in Mathematics, Physics, and Maritime Systems

---

# ⭐ If You Found This Project Interesting

Feel free to:
- Star the repository
- Fork the project
- Suggest improvements
- Connect for collaboration

---

# 📬 Contact

📧 **Email:** med.marin17@gmail.com  
🔗 **LinkedIn:** https://www.linkedin.com/in/mohammed-e-a13664182  
🐙 **GitHub:** https://github.com/IBOSS24  
