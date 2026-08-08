# AI/ML Internship Journey 🤖

A weekly collection of projects and implementations covering **Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, and Neural Networks**.

The repository documents the progression from fundamental AI/ML concepts to practical applications such as object detection, image search, and image classification.

---

# Week 1 - TensorFlow Fundamentals 🧠

## Overview

This week focuses on the fundamentals of TensorFlow and building a basic neural network for handwritten digit classification using the **MNIST dataset**.

## 📚 Topics Covered

* TensorFlow installation and environment setup
* Tensor basics
* Scalars, vectors, matrices, and 3D tensors
* Tensor shapes and data types
* Tensor reshaping
* Basic TensorFlow operations
* Neural network fundamentals
* MNIST handwritten digit classification
* Model training and evaluation
* Saving trained TensorFlow models
* Generating predictions

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

## 📁 Project Structure

```text
Week-1-TensorFlow/
│
├── models/
│   └── mnist_model.keras
│
├── outputs/
│   └── mnist_prediction.png
│
├── main.py
├── tensor_basics.py
├── mnist_classifier.py
├── requirements.txt
└── README.md
```

---

# Week 2 - Object Detection 🎯

## Overview

Implemented an object detection system using **OpenCV and YOLO** to identify objects in images and generate bounding boxes with confidence scores.

## ✨ Features

* OpenCV-based object detection
* YOLO-based object detection
* Bounding box generation
* Confidence score display
* Detection of multiple objects

## 🤖 Model

* **YOLO11 Nano (`yolo11n.pt`)**
* Pre-trained object detection model

## 📊 Result

The system successfully detected objects such as:

* Bus
* Person

The model achieved up to **94.02% confidence** for the detected bus.

## 📁 Project Structure

```text
Week-2-Object-Detection/
│
├── images/
├── outputs/
├── object_detection.py
├── yolo_detection.py
├── requirements.txt
└── README.md
```

---

# Week 3 - Advanced Object Detection 🚀

## Overview

Implemented an advanced object detection and tracking pipeline using **YOLO** for processing video-based inputs.

## ✨ Features

* YOLO-based object detection
* Object tracking
* Bounding box generation
* Confidence score detection
* Video-based processing

## 🤖 Model

* **YOLO11 Nano (`yolo11n.pt`)**
* Pre-trained object detection model

## 📊 Result

The YOLO model was successfully loaded and the advanced detection pipeline was implemented.

> **Note:** Video tracking could not be executed because OpenCV was unable to open the provided video file.

## 📁 Project Structure

```text
Week-3-Advanced-Object-Detection/
│
├── videos/
├── outputs/
├── advanced_detection.py
├── requirements.txt
└── README.md
```

---

# Week 4 - Product Image Search 🔍

## Overview

Implemented a product image search system that compares a **query image** with a collection of product images and returns the most visually similar products.

## ✨ Features

* Image feature extraction
* Product image similarity comparison
* Top similar product retrieval
* Similarity score generation
* Image-based product search

## 📊 Result

The system successfully identifies and ranks similar product images based on their extracted visual features.

## 📁 Project Structure

```text
Week-4-Product-Image-Search/
│
├── images/
├── outputs/
├── product_search.py
├── requirements.txt
└── README.md
```

---

# Week 5 - Advanced Product Image Search 🔎

## Overview

Implemented an advanced product image search system using **computer vision-based feature extraction and similarity matching**.

The system extracts visual features from product images and identifies the most similar products using **HSV color histograms** and **cosine similarity**.

## ✨ Features

* Product image feature extraction
* HSV color histogram generation
* Feature vector comparison
* Cosine similarity matching
* Top-K similar product retrieval
* Fast image-based product search

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-Learn

## ⚙️ Working

```text
Query Image
     ↓
Image Preprocessing
     ↓
HSV Color Conversion
     ↓
Feature Extraction
     ↓
Feature Vector Generation
     ↓
Cosine Similarity
     ↓
Top-K Similar Products
```

## 📁 Project Structure

```text
Week-5-Advanced-Product-Search/
│
├── images/
│   ├── shoe1.jpg
│   ├── shoe2.jpg
│   ├── shoe3.jpg
│   ├── bag1.jpg
│   └── watch1.jpg
│
├── outputs/
├── advanced_product_search.py
├── requirements.txt
└── README.md
```

---

# Week 6 - Image Classification 👕

## Overview

Implemented an image classification system using a **Convolutional Neural Network (CNN)** trained on the **Fashion-MNIST dataset**.

The model classifies clothing images into **10 different categories** using TensorFlow and Keras.

## ✨ Features

* Fashion-MNIST dataset loading
* Image normalization and preprocessing
* CNN-based image classification
* Model training and validation
* Test set evaluation
* Sample image predictions
* Trained model saving

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy

## 📊 Dataset

The project uses the **Fashion-MNIST** dataset containing grayscale images of clothing items.

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

### Dataset Configuration

For lightweight training, the project uses:

* **10,000 training images**
* **2,000 testing images**
* Image size: **28 × 28 pixels**
* Grayscale images

## 🧠 CNN Architecture

```text
Input (28 × 28 × 1)
        ↓
Conv2D (32 filters)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128 neurons)
        ↓
Dense (10 classes)
```

## 📁 Project Structure

```text
Week-6-Image-Classification/
│
├── outputs/
├── image_classifier.py
├── requirements.txt
└── README.md
```

---

# Week 7 - Advanced Image Classification 🚀

## Overview

Implemented an advanced image classification pipeline using a **Convolutional Neural Network (CNN)** with **data augmentation, batch normalization, and dropout regularization**.

The model is trained on the **Fashion-MNIST dataset** to classify clothing images into 10 categories.

## ✨ Features

* Fashion-MNIST dataset loading
* Image normalization
* Data augmentation
* Deep CNN architecture
* Batch Normalization
* Dropout regularization
* Model training and validation
* Test set evaluation
* Sample predictions
* Trained model saving

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy

## 📊 Dataset

The project uses the **Fashion-MNIST** dataset containing grayscale images of clothing items.

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

### Dataset Configuration

For lightweight training, the project uses:

* **10,000 training images**
* **2,000 testing images**
* Image size: **28 × 28 pixels**
* Grayscale images

## 🧠 Model Architecture

```text
Input (28 × 28 × 1)
        ↓
Data Augmentation
        ↓
Conv2D (32 filters)
        ↓
Batch Normalization
        ↓
MaxPooling
        ↓
Conv2D (64 filters)
        ↓
Batch Normalization
        ↓
MaxPooling
        ↓
Conv2D (128 filters)
        ↓
Batch Normalization
        ↓
Flatten
        ↓
Dense (128 neurons)
        ↓
Dropout (0.4)
        ↓
Dense (10 classes)
```

## 📁 Project Structure

```text
Week-7-Advanced-Image-Classification/
│
├── outputs/
├── advanced_classifier.py
├── requirements.txt
└── README.md
```

---

# 🏁 Progress

| Week   | Project                       | Main Concept                           |
| ------ | ----------------------------- | -------------------------------------- |
| Week 1 | TensorFlow Fundamentals       | TensorFlow & MNIST                     |
| Week 2 | Object Detection              | YOLO & OpenCV                          |
| Week 3 | Advanced Object Detection     | Detection & Tracking                   |
| Week 4 | Product Image Search          | Image Similarity                       |
| Week 5 | Advanced Product Search       | Feature Extraction & Cosine Similarity |
| Week 6 | Image Classification          | CNN & Fashion-MNIST                    |
| Week 7 | Advanced Image Classification | Augmentation, BatchNorm & Dropout      |

---

## 🎯 Learning Progression

```text
TensorFlow Fundamentals
          ↓
Object Detection
          ↓
Advanced Object Detection
          ↓
Product Image Search
          ↓
Advanced Product Search
          ↓
Image Classification
          ↓
Advanced Image Classification
```

This repository documents the progression from **TensorFlow fundamentals to advanced computer vision and deep learning techniques**.
