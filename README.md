
# Iris Flower Classification with Support Vector Machine (SVM)

Train and evaluate a Support Vector Machine model for classifying Iris flower species.

![Iris Flowers](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Author](#author)
- [License](#license)

## Overview

This project demonstrates how to train a Support Vector Machine (SVM) model to classify Iris flower species using the popular Iris dataset. It consists of two main scripts: `train_and_save_model.py`, which trains an SVM model and saves it to a file, and `load_and_evaluate_model.py`, which loads the trained model and evaluates its performance.

## Installation

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your_username/iris-flower-classification.git
   ```

2. Navigate to the project directory:
   ```
   cd iris-flower-classification
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the SVM model and save it:
   ```
   python train_and_save_model.py
   ```

2. Load the trained model and evaluate its performance:
   ```
   python load_and_evaluate_model.py
   ```

## File Descriptions

1. **train_and_save_model.py**: Script to load the Iris dataset, preprocess the data, train an SVM model, and save the trained model to a file.

2. **load_and_evaluate_model.py**: Script to load the saved SVM model, preprocess the test data, evaluate the model's performance, and plot decision boundaries.

3. **iris_svm_model.pkl**: File containing the saved SVM model.

4. **requirements.txt**: File containing the required Python packages for running the scripts.

## Requirements

- Joblib (==1.1.0)
- Matplotlib (==3.5.1)
- NumPy (==1.21.3)
- Scikit-learn (==1.0.2)

## Author

[Your Name/Username]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
