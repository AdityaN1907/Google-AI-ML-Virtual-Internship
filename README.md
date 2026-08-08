# Week 1 - TensorFlow Fundamentals 🧠

This week focuses on the fundamentals of TensorFlow and building a basic neural network for handwritten digit classification using the MNIST dataset.

## 📚 Topics Covered

- TensorFlow installation and environment setup
- Tensor basics
- Scalars, vectors, matrices and 3D tensors
- Tensor shapes and data types
- Tensor reshaping
- Basic TensorFlow operations
- Neural network fundamentals
- MNIST handwritten digit classification
- Model training and evaluation
- Saving trained TensorFlow models
- Generating predictions

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

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

# Week 2 - Object Detection

## Overview
Implemented object detection using OpenCV and YOLO to identify objects and generate bounding boxes.

## Features
- OpenCV-based object detection
- YOLO-based object detection
- Bounding box generation
- Confidence score display

## Model
- YOLO11 Nano (`yolo11n.pt`)
- Pre-trained object detection model

## Result
Detected objects including:
- Bus
- Person

Example detection achieved up to **94.02% confidence** for the bus.

## Project Structure

```text
Week-2-Object-Detection/
├── images/
├── outputs/
├── object_detection.py
├── yolo_detection.py
├── requirements.txt



# Week 3 - Advanced Object Detection

## Overview
Implemented advanced object detection and object tracking using YOLO.

## Features
- YOLO-based object detection
- Object tracking
- Bounding box generation
- Confidence score detection
- Video-based processing

## Model
- YOLO11 Nano (`yolo11n.pt`)
- Pre-trained detection model

## Result
The YOLO model was successfully loaded and the advanced detection pipeline was implemented.

> Note: Video tracking could not be executed because OpenCV was unable to open the provided video file.

## Project Structure

```text
Week-3-Advanced-Object-Detection/
├── videos/
├── outputs/
├── advanced_detection.py
├── requirements.txt




# Week 4 - Product Image Search

## Overview
Implemented a product image search system that compares a query image with a collection of product images and returns the most similar products.

## Features
- Image feature extraction
- Product image similarity comparison
- Top similar product retrieval
- Similarity score generation

## Result
The system successfully identifies and ranks similar product images based on their extracted features.

## Project Structure

```text
Week-4-Product-Image-Search/
├── images/
├── outputs/
├── product_search.py
├── requirements.txt


# Week 5 - Advanced Product Image Search 

## Overview

This project demonstrates an advanced product image search system using computer vision techniques. The application extracts visual features from product images and identifies the most similar products based on feature similarity.

The system uses OpenCV for image processing and Scikit-Learn for cosine similarity matching.

---

## Features

- Product image feature extraction
- HSV color histogram generation
- Feature vector comparison
- Cosine similarity matching
- Top-K similar product retrieval
- Fast image-based product search

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Learn

---

## Project Structure

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
│
├── advanced_product_search.py
├── requirements.txt




# Week 6 - Image Classification 

## Overview

This project demonstrates image classification using a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset.

The model classifies clothing images into 10 different categories using TensorFlow and Keras.

---

## Features

- Fashion-MNIST dataset loading
- Image normalization and preprocessing
- CNN-based image classification
- Model training and validation
- Test set evaluation
- Class prediction for sample images
- Trained model saving

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy

---

## Dataset

The project uses the **Fashion-MNIST** dataset containing grayscale images of clothing items.

### Classes

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

For lightweight training, the project uses:

- **10,000 training images**
- **2,000 testing images**
- Image size: **28 × 28 pixels**

---

## CNN Architecture

The model consists of:

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





# Week 7 - Advanced Image Classification 

## Overview

This project demonstrates an advanced image classification pipeline using a Convolutional Neural Network (CNN) with data augmentation, batch normalization, and dropout.

The model is trained on the Fashion-MNIST dataset to classify clothing images into 10 categories.

---

## Features

- Fashion-MNIST dataset loading
- Image normalization
- Data augmentation
- Deep CNN architecture
- Batch Normalization
- Dropout regularization
- Model training and validation
- Test set evaluation
- Sample predictions
- Trained model saving

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy

---

## Dataset

The project uses the Fashion-MNIST dataset containing grayscale images of clothing items.

### Classes

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

For lightweight training, the project uses:

- **10,000 training images**
- **2,000 testing images**
- Image size: **28 × 28 pixels**

---

## Model Architecture

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