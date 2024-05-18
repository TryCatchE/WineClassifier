# Wine Quality Classifier

This repository contains code for building and evaluating classifiers to predict the quality of wine based on various features.

## Description

This project includes two sets of code snippets:

1. **Set 1**: Utilizes scikit-learn for building a classification model.
2. **Set 2**: Utilizes Keras with TensorFlow backend for building a neural network-based classification model.

## Set 1: Scikit-Learn Classifier

### Features

- Utilizes scikit-learn for preprocessing, model training, and evaluation.
- Implements a multi-layer perceptron classifier.
- Includes functions for plotting confusion matrix, learning curve, class distribution, and correlation matrix.
- Provides functions for loading data, splitting, preprocessing, training, and evaluating the model.

### Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
- scipy

### Usage

1. Ensure the required packages are installed.
2. Run the `nn()` the to execute the code.
3. The script loads the dataset, preprocesses it, trains the classifier, and evaluates its performance.
4. The classification report, accuracy, and best parameters are printed, along with visualizations of class distribution and correlation matrix.

## Set 2: Keras Neural Network Classifier

### Features

- Utilizes Keras with TensorFlow backend for building a neural network-based classifier.
- Implements data preprocessing, model training, and evaluation.
- Includes functions for computing class weights, plotting boxplots, class distribution, confusion matrix, and correlation matrix.
- Provides functions for data preprocessing, model creation, training, and evaluation.

### Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn
- TensorFlow
- Keras

### Usage

1. Ensure the required packages are installed.
2. Run the `tensorNN.py` the to execute the code.
3. The script loads the dataset, preprocesses it, trains the neural network, and evaluates its performance.
4. The classification report, accuracy, and confusion matrix are printed, along with visualizations of class distribution and correlation matrix.
5. Predictions are made on new data samples, and their corresponding quality levels are displayed.

