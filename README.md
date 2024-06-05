# Plant_Disease_Prediction
# Malaria Detection using Convolutional Neural Network (CNN)

This project demonstrates a deep learning approach to detect malaria in cell images using a Convolutional Neural Network (CNN). The model is trained to distinguish between healthy and malaria-infected cell images, leveraging data augmentation and visualization techniques to enhance performance and interpret results.

## Features

- **Data Augmentation**: Applies transformations such as rescaling, shearing, zooming, and horizontal flipping to improve the robustness of the model.
- **CNN Architecture**: A sequential neural network with convolutional, pooling, flattening, and dense layers to extract and learn features from images.
- **Binary Classification**: Classifies images as either healthy or infected with malaria.
- **Real-time Visualization**: Plots training/validation accuracy and loss, displays confusion matrix, and compares the model's accuracy with state-of-the-art models.

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Pillow)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/malaria-detection-cnn.git
    cd malaria-detection-cnn
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn pillow
    ```

## Usage

1. **Dataset Preparation**:
    - Organize your dataset into two main directories: `training_set` and `testing_set`.
    - Each directory should have two subdirectories: `healthy` and `unhealthy`, containing respective images.

2. **Run the Script**:
    ```bash
    python malaria_detection_cnn.py
    ```

## Model Overview

### Architecture

The CNN architecture consists of multiple layers:
- **Convolutional Layers**: Extract features from the input images.
- **Pooling Layers**: Reduce the spatial dimensions of the feature maps.
- **Flatten Layer**: Converts the 2D feature maps to a 1D feature vector.
- **Dense Layers**: Perform classification based on the extracted features.
- **Output Layer**: Uses sigmoid activation for binary classification.

### Data Augmentation

The model employs data augmentation techniques to enhance generalization by artificially increasing the diversity of the training dataset.

### Training and Evaluation

The model is trained on augmented images from the training set and validated on the testing set. Training history is recorded to plot the accuracy and loss over epochs.

### Visualizations

1. **Training & Validation Accuracy and Loss**:
    - The plots show how the model's accuracy and loss change over epochs for both the training and validation sets.

    ![image](https://github.com/mostafaelesely/Plant_Disease_Prediction/assets/138331364/c9227f87-e512-4385-8efe-5fc0c354665b)

   ![image](https://github.com/mostafaelesely/Plant_Disease_Prediction/assets/138331364/f162eae5-54d7-4a41-9219-1dceb0bf40a6)



2. **Confusion Matrix**:
    - The confusion matrix visualizes the performance of the model by showing the true positive, true negative, false positive, and false negative counts.

    ![image](https://github.com/mostafaelesely/Plant_Disease_Prediction/assets/138331364/d5418f78-95ee-4883-9df2-7d3b6e28cb01)


3. **Comparison with State-of-the-Art Models**:
    - A bar chart compares the accuracy of the trained model with accuracies of other state-of-the-art models.

   ![image](https://github.com/mostafaelesely/Plant_Disease_Prediction/assets/138331364/729edbcd-d837-4e78-a757-1c889b01a5c5)


## Conclusion

This project demonstrates the use of a Convolutional Neural Network for detecting malaria in cell images. The implementation includes data augmentation, training, evaluation, and various visualizations to help understand the model's performance. Further improvements can be made by experimenting with different architectures, hyperparameters, and additional data augmentation techniques.

