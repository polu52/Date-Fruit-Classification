# Date Fruit Classification Project

This project develops an automated system to classify different types of date fruits using image data. Accurate classification is essential for the agricultural sector to optimize sorting and grading, reducing manual effort and errors. By leveraging Convolutional Neural Networks (CNNs) and Transfer Learning with VGG16, the project aims to enhance efficiency and ensure quality control in the date fruit industry.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Comparison](#comparison)
- [Reflection](#reflection)
- [Streamlit Web Application](#streamlit-web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
The project focuses on classifying various types of date fruits using image data. This automation helps in reducing manual errors and effort in the agricultural sector, particularly in sorting and grading processes. The project employs both a custom CNN model and Transfer Learning with the VGG16 model for classification.

## Dataset
The dataset comprises labeled images of different date fruit types. Key data preprocessing steps include:
- **Image Standardization**: All images are resized to 170x170 pixels (RGB).

## Model Architecture

### 1. Custom CNN Model:
- **Architecture**: Multiple Conv2D and MaxPooling2D layers were used for feature extraction, followed by Flatten and Dense layers for classification.
- **Activation Functions**: ReLU was used in hidden layers, and softmax for the output layer.
- **Training**: The model was trained using the categorical crossentropy loss function, Adam optimizer, with regularization techniques such as dropout and early stopping to avoid overfitting.

### 2. Transfer Learning with VGG16:
- **Base Model**: VGG16 pretrained on ImageNet with frozen convolutional layers.
- **Top Layers**: Flatten, Dense, and Dropout layers were added to suit the date fruit classification task.
- **Training**: The top layers were finetuned using the same loss function and optimizer

## Results

### Custom CNN Model:
- **Training Accuracy**: 99%
- **Validation Accuracy**: 91%

### VGG16 Transfer Learning:
- **Training Accuracy**: 85%
- **Validation Accuracy**: 90%

## Comparison
Both the custom CNN and VGG16 models performed similarly in validation accuracy. However, the VGG16 model is less prone to overfitting and could achieve even better results with more epochs. The CNN model had slightly higher training accuracy but showed signs of overfitting compared to the more generalized performance of the VGG16 model.

## Reflection
This project highlighted the advantages of using Transfer Learning over a custom CNN model for agricultural image classification tasks. Key takeaways include:
- **Transfer Learning Benefits**: Improved accuracy and generalization with less training time.
- **Data Quality**: Ensuring high-quality, diverse data is essential for achieving good model performance, particularly for real-world applications.
- **Regularization**: Techniques such as dropout and early stopping are crucial in preventing overfitting, especially in custom-built models.

### Future Directions:
- Further exploration of advanced data augmentation techniques.
- Hyperparameter tuning for improved performance.
- Exploration of other pretrained models like ResNet for comparison.
- Improved validation strategies to ensure robustness in real-world scenarios.

## Streamlit Web Application
A web application was created using Streamlit, allowing users to upload images of date fruits and receive predictions on their classification.

You can try the web application hosted on Hugging Face Spaces:
👉 [Date Fruit Classification Web App](https://huggingface.co/spaces/poluhamdi/datefruit)

Explore the project's code on Kaggle:
👉 [Kaggle Notebook](https://www.kaggle.com/code/hamdipolu/date-fruit-classification/notebook)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/date-fruit-classification.git
    cd date-fruit-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `data/` folder.

## Usage
To run the Streamlit web application locally, use the following command:
```bash
streamlit run app.py
```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
