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


🔢 MNIST Digit Classifier

A neural network was trained using the MNIST handwritten digit dataset.

Dataset
Training images: 60,000
Testing images: 10,000
Image size: 28 × 28 pixels
Number of classes: 10 (digits 0–9)
Pixel values normalized to the range 0–1
Model Architecture
Input Image (28 × 28)
        ↓
Flatten
        ↓
Dense Layer (128 neurons)
        ↓
Dense Layer (10 neurons)
        ↓
Digit Prediction (0–9)
Training
Epochs: 5
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Results
Metric	Result
Training Accuracy	98.54%
Validation Accuracy	97.72%
Test Accuracy	97.64%
Test Loss	0.0771

The trained model was successfully saved as:

models/mnist_model.keras

A prediction visualization was also generated:

outputs/mnist_prediction.png
▶️ How to Run

Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

Run TensorFlow basics:

python main.py

Run tensor basics:

python tensor_basics.py

Run the MNIST classifier:

python mnist_classifier.py
🎯 Learning Outcome

By completing this week, I gained hands-on experience with TensorFlow tensors, basic neural network architecture, model training, evaluation, prediction and model saving.