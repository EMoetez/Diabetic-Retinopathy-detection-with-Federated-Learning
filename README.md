# Diabetic-Retinopathy-detection-with-Federated-Learning

This project explores a federated learning approach for detecting diabetic retinopathy (DR) from retina images. It enhances the accuracy of multi-class classification while ensuring data privacy across distributed healthcare institutions.

## Project Overview

Diabetic retinopathy is a vision-threatening condition affecting people with diabetes. This approach integrates **Convolutional Neural Networks (CNNs)** and **Federated Learning** to enable collaborative learning without compromising data privacy. The project includes:
- Multi-class classification model trained on retina images
- Comparison between centralized and federated learning approaches
- Pretrained CNNs including VGG16, VGG19, and ResNet50
- Evaluation of model performance in federated setups

## Methodology

### 1. Data Exploration and Preprocessing
   - **Dataset**: Retina images labeled by DR severity levels (e.g., No DR, Mild, Moderate, Proliferative, and Severe).
   - **Data Augmentation**: Synthetic sample generation via transformations (e.g., rotation, flip).
   - **Preprocessing**: Applied contrast enhancement and denoising for optimal model input.

### 2. Model Selection and Training
   - Custom CNN model for baseline testing
   - Comparison of pretrained models: VGG16, VGG19, and ResNet50
   - Final federated model built on ResNet50 architecture for collaborative training

### 3. Federated Learning Implementation
   - **Data Distribution**: Simulated across 10 federated institutions
   - **Training Process**: Cyclical model updates with local training and global aggregation

## Results

ResNet50 showed the best performance in both centralized and federated setups. Federated learning demonstrated potential for enhanced data privacy without significant loss of accuracy, though it required more rounds for convergence compared to centralized learning.

## Requirements

- Python 3.x
- TensorFlow and Keras
- Kaggle Notebooks (for development environment)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/EMoetez/Diabetic-Retinopathy-detection-with-Federated-Learning.git
   cd Diabetic-Retinopathy-detection-with-Federated-Learning
