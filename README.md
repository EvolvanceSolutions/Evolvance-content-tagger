# Evolvance Content Tagger
### Deep Learning Project 2 of 10 | CNN + Transfer Learning

A production-ready image classification system that automatically categorises
educational content images into 5 categories using MobileNetV2 transfer learning.
Built as part of a 10-project deep learning journey focused on EdTech applications.

---

## What It Does

Upload any educational image and the model instantly classifies it into one of
5 content categories with a confidence score:

| Category | Description |
|---|---|
| Chart | Data visualizations - bar, line, pie charts |
| Diagram | Process diagrams - flowcharts, mindmaps, hierarchies |
| Slide | Presentation slides - title layouts, bullet points |
| Photo | Real-world photographs - classrooms, people, objects |
| Illustration | Digital artwork - flat design, icons, graphics |

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 99.5% |
| Validation Accuracy | 100.0% |
| Training Epochs | 6 (early stopped) |
| Trainable Parameters | 164,613 |
| Total Parameters | 2,422,597 |

---

## Architecture

    Input Image (224x224x3)
           |
    MobileNetV2 Base (frozen, pretrained on ImageNet)
    154 layers, 2,257,984 frozen parameters
           |
    GlobalAveragePooling2D
           |
    Dense(128, ReLU)
           |
    Dropout(0.3)
           |
    Dense(5, Softmax)
           |
    Output: 5 category probabilities

---

## Why Transfer Learning

MobileNetV2 was pretrained by Google on 1.4 million ImageNet images.
Its convolutional layers already detect edges, textures, and shapes universally.
By freezing these layers and training only a custom classification head,
we achieve 99.5% accuracy with only 1000 training images and 6 epochs on CPU.
Training from scratch would require 100,000+ images and weeks of GPU time.

---

## Dataset

1000 images across 5 categories (200 per category) generated programmatically:

| Category | Source Method |
|---|---|
| Chart | Generated with matplotlib (bar, line, pie) |
| Diagram | Generated with matplotlib (flowchart, mindmap, hierarchy) |
| Slide | Generated with PIL (title+bullets, two-column, section header) |
| Photo | Programmatically sourced and preprocessed |
| Illustration | Generated with PIL (geometric flat-design style) |

All images resized to 224x224 pixels, normalized to 0-1 range.
Train/test split: 80/20 with fixed random seed for reproducibility.

---

## Key Concepts Demonstrated

- Convolutional Neural Networks (CNN)
- Transfer Learning with MobileNetV2
- Feature map extraction and GlobalAveragePooling
- Dropout regularization to prevent overfitting
- EarlyStopping and ModelCheckpoint callbacks
- Softmax multi-class classification
- Categorical cross-entropy loss
- Synthetic dataset generation for controlled experiments
- Data normalization and preprocessing pipelines
- Model serialization and deployment

---

## Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow 2.x / Keras | Model building and training |
| MobileNetV2 | Pretrained base model |
| NumPy | Array operations |
| Pillow (PIL) | Image generation and preprocessing |
| Matplotlib | Chart and diagram generation |
| Streamlit | Web application interface |
| Python 3.10 | Core language |

---

## Project Structure

    Evolvance-content-tagger/
    |
    |- app_p02.py                  Streamlit web application
    |- p02_content_tagger.keras    Trained model weights
    |- p02_categories.npy          Category label mapping
    |- project02_results.png       Confusion matrix and accuracy plots
    |- requirements.txt            Python dependencies
    |- README.md                   This file

---

## How to Run Locally

1. Clone the repository

    git clone https://github.com/EvolvanceSolutions/Evolvance-content-tagger.git
    cd Evolvance-content-tagger

2. Create environment and install dependencies

    conda create -n dlprojects python=3.10 -y
    conda activate dlprojects
    pip install -r requirements.txt

3. Run the web app

    streamlit run app_p02.py

---

## How to Use the Web App

1. Open the app at localhost:8501
2. Click Upload and select any PNG or JPG image
3. The model classifies it instantly with confidence scores
4. Each prediction includes a suggested LMS action

---

## Results

The confusion matrix shows perfect or near-perfect separation across all
5 categories. The only marginal confusion (0.3%) occurs between photo and
illustration -- both categories share organic, non-geometric visual patterns.
This is consistent with human visual perception of these two categories.

---

## What I Learned Building This

- Transfer learning reduces training data requirements by 99% compared to
  training from scratch
- Synthetic data generation is a valid approach when labeled real-world
  data is unavailable or expensive to collect
- EarlyStopping at epoch 6 prevented overfitting
- GlobalAveragePooling2D outperforms Flatten for small datasets
- MobileNetV2's 7x7x1280 output perfectly separates visually distinct
  educational content categories

---

## Part of a 10-Project Deep Learning Series

| Project | Topic | Status |
|---|---|---|
| 01 | Neural Network from Scratch - Student Score Predictor | Complete |
| 02 | CNN - Evolvance Content Tagger | Complete |
| 03 | LSTM - Student Feedback Sentiment Analyzer | Up next |
| 04 | DNN - Student Dropout Predictor | Planned |
| 05 | Transformer - Auto Quiz Generator | Planned |
| 06 | Whisper + BART - Lecture Transcriber | Planned |
| 07 | CNN + CTC - Handwriting Recognition | Planned |
| 08 | Autoencoder - Learning Path Recommender | Planned |
| 09 | RAG - AI Tutor Chatbot | Planned |
| 10 | Multi-modal - Course Quality Analyzer | Planned |

---

## Author

Built by Evolvance
EdTech and Agentic AI specialists
https://github.com/EvolvanceSolutions
