# Diabetic-Retinopathy-detection-with-Federated-Learning
This project explores a federated learning approach for detecting diabetic retinopathy (DR) from retina images. It enhances the accuracy of multi-class classification while ensuring data privacy across distributed healthcare institutions.

![Screenshot_20240207_061832](https://github.com/user-attachments/assets/01724a18-7bfd-4e7a-9185-4c07114927d4)

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#Project-Overview)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Requirements](#Requirements)
6. [Furure work](#Future-work)
7. [License](#license)

## Installation

### Step 1: Clone the repository
   ```bash
   git clone https://github.com/EMoetez/Diabetic-Retinopathy-detection-with-Federated-Learning.git
   cd Diabetic-Retinopathy-detection-with-Federated-Learning
   ```
### Step 2: Install Dependencies
Install all required dependencies:
```bash
pip install -r requirements.txt
```


## Project Overview

Diabetic retinopathy is a vision-threatening condition affecting people with diabetes. This approach integrates **Convolutional Neural Networks (CNNs)** and **Federated Learning** to enable collaborative learning without compromising data privacy. The project includes:
- Multi-class classification model trained on retina images
- Comparison between centralized and federated learning approaches
- Pretrained CNNs including VGG19 and ResNet50
- Evaluation of model performance in federated setups

## Methodology

### 1. Data Exploration and Preprocessing
   - **Dataset**: Retina images labeled by DR severity levels (e.g., No DR, Mild, Moderate, Proliferative, and Severe).
   - **Data Augmentation**: Synthetic sample generation via transformations (e.g., rotation, flip).
   - **Preprocessing**: Applied contrast enhancement and denoising for optimal model input.
 <br/><br/>
     ![Screenshot_20240329_012438](https://github.com/user-attachments/assets/441d9184-ef08-47df-b014-d80da7fdff66)
![Screenshot_20240210_085956](https://github.com/user-attachments/assets/9ef26218-da06-4ac5-b01c-cfbfa44f57e3)

### 2. Model Selection and Training
   - Custom CNN model for baseline testing
   - Comparison of pretrained models: VGG19, and ResNet50
   - Final federated model built on ResNet50 architecture for collaborative training
     <br/><br/>
     ![RESNET50RESULTS](https://github.com/user-attachments/assets/4a962c61-4380-4a5e-af53-33dd7ee91016)


### 3. Federated Learning Implementation
   - **Data Distribution**: Simulated across multiple federated institutions
   - **Training Process**: Cyclical model updates with local training and global aggregation
     <br/><br/>
     ![FL_approach](https://github.com/user-attachments/assets/6a4ee531-5f83-487f-a8a6-49df59daa719)


## Results

ResNet50 showed the best performance in both centralized and federated setups. Federated learning demonstrated potential for enhanced data privacy without significant loss of accuracy, though it required more rounds for convergence compared to centralized learning.

## Requirements

- Python 3.x
- TensorFlow and Keras
- Kaggle Notebooks (for development environment)

## Future work
In the next steps, we aim to improve the results, optimize the code and use different dataset like "APTOS 2019" dataset. In addition, we will work on implementing the federated learning process with more lightweight and friendly frameworks like Flower. This way, we hope to get faster and more accurate results.

## License
This project is licensed under the MIT License.


