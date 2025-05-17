# BrainTumorClassifier
CNN model for classifying brain tumor types using MRI scans
# 🧠 Brain Tumor Classifier

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into different types of brain tumors.

## 📊 Goal
To assist in the early detection and classification of brain tumors using deep learning techniques applied to MRI scans.

## 📁 Dataset
The dataset contains labeled MRI images of brain tumors from different classes such as:
- **Pituitary**
- **Meningioma**
- **Glioma**

Each image is grayscale and standardized in size. The data was split into training and testing sets.

## 🧠 Model Architecture
A custom CNN model was trained from scratch using:
- Convolutional layers
- Max pooling
- ReLU activations
- Fully connected layers
- Softmax for multi-class classification

The model was implemented using **PyTorch**.

## 📈 Results
- **Accuracy:** 92.7% on a test set of 1000 images
- **Prediction Log:** A CSV file (`predictions_log.csv`) contains prediction results for manual inspection

## 📂 Project Structure
