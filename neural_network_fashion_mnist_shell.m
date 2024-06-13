clear;
close all;
clc;

% This is a shell script that gives a structure that can be used to train a NN on the FASHION MNIST
% dataset. Many lines and some functions are not filled in yet. Finish this
% script to be able to train your neural network and produce the requested
% deliverables. Provide additional comments to your code.

% Load Fashion MNIST data
run('Fashion_MNIST.m');

X_train = reshape(training.images, [size(training.images,1)*size(training.images,2), size(training.images,3)]);
X_test = reshape(test.images, [size(test.images,1)*size(test.images,2), size(test.images,3)]);
Y_train = my_onehotencode(training.labels, 0, 9); % FUNCTION AT BOTTOM OF SCRIPT
Y_test = my_onehotencode(test.labels, 0, 9);

%% Initializing the layers of the neural network

% hidden layer 1 784 inputs; 300 outputs
W1 = normrnd(0,sqrt(1/784),300,784);
b1 = normrnd(0,1,300,1);

% hidden layer 2 300 inputs; 100 outputs
W2 = normrnd(0,sqrt(1/300),100,300);
b2 = normrnd(0,1,100,1);

% hidden layer 3 100 inputs; 10 outputs
W3 = normrnd(0,sqrt(1/100),10,100);
b3 = normrnd(0,1,10,1);

%% Train settings
batch_size = 100;
learning_rate = 0.1;
no_epochs = 100;

%% Perform the training
pre_factor = learning_rate * (1/batch_size); % We need this a lot so compute only once outside the loop    

% To keep track of the progress we store the losses after each epoch in a list for later analysis
losses_train = zeros(1,no_epochs);
losses_test = zeros(1,no_epochs);

for epoch=1:no_epochs
    % For each epoch, train over all the batches
    batch_nr = 1;
    sample_size = size(X_train,2); 
    while batch_nr*batch_size <= sample_size
        [X_batch, Y_batch] = batch(X_train,Y_train,batch_size,batch_nr); % FUNCTION AT THE BOTTOM OF SCRIPT
        
        % forward pass
        a0 = X_batch;
        z1 = W1 * a0 + b1;
        a1 = sigmoid(z1);       
        z2 = ... 
        a2 = ...           % Sigmoid (implement at bottom of the script)
        z3 = ...
        a3 = ...           % Softmax (implement at bottom of the script
        
        % backward pass
        dz3 = a3 - Y_batch;
        dz2 = ...
        dz1 = ...

        % gradient descent step / update weights and biases  
        W1 = W1 - pre_factor * dz1 * a0';
        W2 = ...
        W3 = ...
        
        b1 = b1 - pre_factor * sum(dz1,2);
        b2 = ...
        b3 = ...
        
        batch_nr = batch_nr + 1;
    end
    
    % for each epoch evaluate the network performance
    % compute accuracy and the loss of both train- and testset
    a0 = X_train;
    a1 = ...
    a2 = ...
    a3 = ...
    
    acc_train = 100 * accuracy(Y_train,a3); % FUNCTION AT BOTTOM OF SCRIPT
    ln = log(a3);
    loss_train = -mean(ln(Y_train==1));  % y is hot-one encoded so only terms where y=1 need to be considered
    losses_train(epoch) = loss_train;

    % Repeat for the test data:
    a0 = ...
    a1 = ...
    a2 = ...
    a3 = ...
    
    acc_test = ...
    ln = ...
    loss_test = ...  % y is hot-one encoded so only terms where y=1 need to be considered
    losses_test(epoch) = loss_test;

    report_text = 'Epoch nr %d|%d Train Accuracy: %.1f Test Accuracy %.1f Train loss: %.2e Test loss: %.2e \n';
    fprintf(report_text,epoch,no_epochs,acc_train,acc_test,loss_train,loss_test)
end

%% Plotting losses
figure;semilogy(1:no_epochs,losses_train,1:no_epochs,losses_test);
xlabel('iteration');ylabel('CE loss');legend('training set','testing set');grid on

%% Check the prediction of 10 random images
figure;tiledlayout(2,5)
for i=1:10
    rand_image = randi(size(X_test,2));
    a0 = X_test(:,rand_image);
    a1 = ...
    a2 = ...
    a3 = ...
    prediction = ...
    nexttile
    image(test.images(:,:,rand_image)*255)
    title(['Prediction: ', int2str(prediction)])
end

%% Functions

function acc = accuracy(Y,Y_prob)
% Returns accuracy of probability vectors against hot encoded values, as a value between 0 and 1
% Y - the one-hot-encoded labels of each image (columns)
% Y_prob the output vectors of the NN corresponding to the images
% (columns). These can be interpreted as probabilities.
    correct = 0;
    for i=1:size(Y,2)
        [~,max_index_Y] = max(Y(:,i));
        [~,max_index_Y_prob] = max(Y_prob(:,i));
        if max_index_Y == max_index_Y_prob
            correct = correct + 1;
        end
    end
    acc = correct/size(Y,2);
end

function [X_batch, Y_batch] = batch(X,Y,batch_size,i)
% Returns the i'th batch of X and Y given the batch size
% X - complete set of input data
% Y - complete set of one-hot-encoded labels
% batch_size - size of batch to be returned (nr of images to assess in a
% batch)
% i - batch index. For example, a batch size of 20 and an index of 3 will
% return element 41-60
    j = (i-1)*batch_size + 1;
    batch_interval = [j, j + batch_size - 1];
    sample_size = size(X,2);
    if batch_interval(2) > sample_size
        batch_interval(2) = sample_size;
    end
    X_batch = X(:,batch_interval(1):batch_interval(2));
    Y_batch = Y(:,batch_interval(1):batch_interval(2));
end

function Y_encoded = my_onehotencode(Y,min,max)
% Returns a matrix with row wise stacking of one-hot-encoding of the contents of Y given 
% a certain minimum value and maximum value to be encoded
    ...
end

function s = sigmoid(z)
% Sigmoid function
    s = ...
end

function diff = sigmoid_diff(z)
% Derivative of the sigmoid function
    s = ...
    diff = ...
end

function res = softmax(z)
% Softmax function
    exp_z = ...
    res = ...
end