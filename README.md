## Neural Network Fashion MNIST Analysis

### Group 30: Morgan Arima and Jennifer Dieltiens

This repository contains scripts and analysis for the Neural Network Fashion MNIST project.

### Primary Goals Achieved:

a) Plot the accuracy (% of correct classifications) of the Neural Network model on the test and train set versus the number of epochs.

b) Subplot of 10 random images from the test set together with predicted labels.

c) Hyperparameter tuning of batch size and learning rate. Plots of loss versus epoch number for various hyperparameters. Explanation of optimal parameters.

### Results Summary:

#### Hyperparameter Tuning:

The most optimal parameters found are:
- Learning Rate: 0.1
- Batch Size: 256

#### Topology Analysis:

Different NN topologies were analyzed, but simpler/smaller networks couldn't achieve the primary goal of 92% accuracy without causing overfitting.

### Advanced Points Achieved:

e) Graph of classification accuracy per fashion item in the test set.

f) Determination of the fashion item with the worst classification accuracy.

g) Histogram of the standard deviations (std) of prediction vectors, split into correctly and incorrectly classified cases.

h) Analysis of using standard deviation of prediction vectors as a signal for classification certainty.

### Excellent Points Achieved:

i) Building the inverse model using an auto-encoder.

j) Training an auto-encoder for compression.

k) Analysis of the latent space using PCA to estimate the number of unique class labels.

### Summary of Scripts:

- `Group30_Arima_Dieltiens.m`: Main script including primary goals and hyperparameter tuning.
- `neural_network_fash_mnist_hyperparam_tuning.m`: Script for hyperparameter tuning.
- `neural_network_fash_minst_optimal_hyperparam_topo_analysis.m`: Analysis of optimal hyperparameters and topologies.
- `neural_network_fashion_mnist_advanced_points.m`: Script for advanced points.
- `auto_encoder_before_compression.m`: Script for building auto-encoder before compression.
- `auto_encoder_after_compression.m`: Script for training auto-encoder after compression.
- `latent_space_analysis.m`: Script for analyzing the latent space using PCA.
- `plottopologies.py`: Python script for plotting topologies.
- `topologiesoutput.txt`: Output file used for plotting topologies.

### Code Repository

All scripts and files related to the project are included in this repository.
