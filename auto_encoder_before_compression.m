clear;
close all;
clc;

% Load Fashion MNIST data using Fashion_MNIST.m script
Fashion_MNIST;

% Extract train and test data from loaded structures
X_train = training.images;
X_test = test.images;

% Display a few examples (optional)
figure;
for i = 1:20
    subplot(4, 5, i);
    imshow(X_train(:, :, i)');
    title(sprintf('Label: %d', training.labels(i)));
end

% Reshape data to be compatible with the network
X_train = reshape(X_train, [28, 28, 1, size(X_train, 3)]);
X_test = reshape(X_test, [28, 28, 1, size(X_test, 3)]);

% Define convolutional autoencoder architecture
encoder_layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10)
    reluLayer
];

decoder_layers = [
    fullyConnectedLayer(192)
    reluLayer
    transposedConv2dLayer(4, 8, 'Stride', 2, 'Cropping', 'same')
    reluLayer
    transposedConv2dLayer(4, 8, 'Stride', 2, 'Cropping', 'same')
    reluLayer
    transposedConv2dLayer(4, 16, 'Stride', 2, 'Cropping', 'same')
    reluLayer
    transposedConv2dLayer(3, 1, 'Stride', 1, 'Cropping', 'same')
    sigmoidLayer
    imageOutputLayer([28 28 1])
];

autoencoder_layers = [
    encoder_layers
    decoder_layers
];

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'Plots', 'training-progress');

% Train the autoencoder
[autoencoder, ~] = trainNetwork(X_train, X_train, autoencoder_layers, options);

% Reconstruct some images and display them
num_images = 10;
figure;
tiledlayout(2, num_images);
for i = 1:num_images
    % Select a random test image
    idx = randi(size(X_test, 4));
    test_image = X_test(:, :, :, idx);
    
    % Reconstruct the image
    reconstructed_image = predict(autoencoder, test_image);
    
    % Plot original and reconstructed images side by side
    nexttile;
    imshow(squeeze(test_image));
    title('Original');
    
    nexttile;
    imshow(squeeze(reconstructed_image));
    title('Reconstructed');
end
