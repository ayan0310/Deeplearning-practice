# Handwritten Digit Classification using TensorFlow & Keras

This project implements a Feed-Forward Neural Network (Artificial Neural Network) to classify handwritten digits from the famous **MNIST dataset**. The model is built using Python and the TensorFlow/Keras framework, achieving an accuracy of approximately **98%** on the test dataset.

## 📌 Project Overview

The goal of this project is to correctly identify digits (0-9) from 28x28 pixel grayscale images. The notebook covers the end-to-end pipeline:
1.  **Data Loading**: Importing the MNIST dataset.
2.  **Preprocessing**: Normalizing pixel values.
3.  **Visualization**: Displaying sample images from the dataset.
4.  **Model Building**: Creating a Sequential Deep Learning model.
5.  **Training**: Training the model for 50 epochs.
6.  **Evaluation**: Calculating accuracy and testing on specific images.

## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow / Keras**: For building and training the neural network.
* **Matplotlib**: For visualizing the handwritten digits.
* **Scikit-Learn**: For calculating accuracy metrics.
* **NumPy**: For numerical array operations.

## ⚙️ Installation & Requirements

To run this notebook, you need a Python environment with the following libraries installed. You can install them via pip:

```bash
pip install tensorflow matplotlib scikit-learn numpy
Note: If you are using Google Colab, these libraries are pre-installed.🚀 How to RunOpen the file Handwritten_digit_classification.ipynb in Jupyter Notebook, Google Colab, or VS Code.Run the cells sequentially.The notebook will automatically download the MNIST dataset on the first run.🧠 Model ArchitectureThe model is a Sequential Artificial Neural Network (ANN) with the following structure:Layer TypeNodes / ShapeActivation FunctionDescriptionFlatten(28, 28) -> 784NoneFlattens 2D image to 1D vectorDense256ReLUHidden Layer 1Dense128ReLUHidden Layer 2Dense32ReLUHidden Layer 3Dense10SoftmaxOutput Layer (Probabilities for digits 0-9)Compilation Details:Optimizer: AdamLoss Function: Sparse Categorical CrossentropyMetrics: Accuracy📊 ResultsTraining Epochs: 50Validation Split: 20%Final Test Accuracy: ~98.15%The model demonstrates high accuracy in distinguishing between different handwritten digits using simple fully connected layers.📝 Example OutputThe notebook includes a prediction step where specific test images are fed into the model.Input Image: * Model Prediction: 2🤝 ContributingFeel free to fork this repository and experiment with Convolutional Neural Networks (CNNs) to see if you can improve the accuracy even further!
