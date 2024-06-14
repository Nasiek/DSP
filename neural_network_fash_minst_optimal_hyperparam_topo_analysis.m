clear;
close all;
clc;

% This script trains a NN on the FASHION MNIST dataset with different topologies

% Load Fashion MNIST data
run('Fashion_MNIST.m');

X_train = reshape(training.images, [size(training.images,1)*size(training.images,2), size(training.images,3)]);
X_test = reshape(test.images, [size(test.images,1)*size(test.images,2), size(test.images,3)]);
Y_train = my_onehotencode(training.labels, 0, 9); % FUNCTION AT BOTTOM OF SCRIPT
Y_test = my_onehotencode(test.labels, 0, 9);

% Define NN topologies
topologies = struct();
topologies.simple = {784, 128, 10};
topologies.medium = {784, 128, 64, 10};
topologies.complex = {784, 256, 128, 64, 10};

% Initialize storage for results
results = struct();

% Train settings
batch_sizes = [32, 64, 128];
learning_rates = [0.01, 0.001, 0.0001];
no_epochs = 100;

for topo_name = fieldnames(topologies)'
    topo_name = topo_name{1};
    layers = topologies.(topo_name);
    
    % Initialize weights and biases for the given topology
    [W, b] = initialize_weights_biases(layers);
    
    for lr = learning_rates
        for bs = batch_sizes
            % Initialize storage for losses and accuracies
            losses_train = zeros(1, no_epochs);
            losses_test = zeros(1, no_epochs);
            accuracies_train = zeros(1, no_epochs);
            accuracies_test = zeros(1, no_epochs);
            
            % Perform the training
            pre_factor = lr * (1/bs);
            sample_size = size(X_train, 2);
            
            for epoch = 1:no_epochs
                batch_nr = 1;
                while batch_nr * bs <= sample_size
                    [X_batch, Y_batch] = batch(X_train, Y_train, bs, batch_nr);
                    
                    % Forward pass
                    [a, z] = forward_pass(X_batch, W, b);
                    
                    % Backward pass
                    [W, b] = backward_pass(X_batch, Y_batch, a, z, W, b, pre_factor);
                    
                    batch_nr = batch_nr + 1;
                end
                
                % Evaluate performance for training data
                [a_train, ~] = forward_pass(X_train, W, b);
                acc_train = 100 * accuracy(Y_train, a_train{end});
                loss_train = compute_loss(Y_train, a_train{end});
                losses_train(epoch) = loss_train;
                
                % Evaluate performance for test data
                [a_test, ~] = forward_pass(X_test, W, b);
                acc_test = 100 * accuracy(Y_test, a_test{end});
                loss_test = compute_loss(Y_test, a_test{end});
                losses_test(epoch) = loss_test;
                
                % Store accuracies
                accuracies_train(epoch) = acc_train;
                accuracies_test(epoch) = acc_test;
                
                % Print progress
                report_text = 'Topology: %s | Epoch: %d/%d | Batch Size: %d | LR: %f | Train Acc: %.1f | Test Acc: %.1f | Train Loss: %.2e | Test Loss: %.2e\n';
                fprintf(report_text, topo_name, epoch, no_epochs, bs, lr, acc_train, acc_test, loss_train, loss_test)
            end
            
            % Store results
            field_name = sprintf('topo_%s_lr_%0.4f_bs_%d', topo_name, lr, bs);
            results.(field_name).losses_train = losses_train;
            results.(field_name).losses_test = losses_test;
            results.(field_name).accuracies_train = accuracies_train;
            results.(field_name).accuracies_test = accuracies_test;
        end
    end
end

% Plotting function
plot_results(results);

%% Function Definitions

function [W, b] = initialize_weights_biases(layers)
    % Initialize weights and biases for given layers
    W = cell(length(layers)-1, 1);
    b = cell(length(layers)-1, 1);
    for i = 1:length(layers)-1
        W{i} = normrnd(0, sqrt(1/layers{i}), layers{i+1}, layers{i});
        b{i} = normrnd(0, 1, layers{i+1}, 1);
    end
end

function [a, z] = forward_pass(X, W, b)
    % Perform forward pass
    a = cell(length(W)+1, 1);
    z = cell(length(W), 1);
    a{1} = X;
    for i = 1:length(W)
        z{i} = W{i} * a{i} + b{i};
        if i == length(W)
            a{i+1} = softmax(z{i});
        else
            a{i+1} = sigmoid(z{i});
        end
    end
end

function [W, b] = backward_pass(X_batch, Y_batch, a, z, W, b, pre_factor)
    % Perform backward pass and update weights and biases
    dz = cell(length(W), 1);
    dz{end} = a{end} - Y_batch;
    for i = length(W)-1:-1:1
        dz{i} = (W{i+1}' * dz{i+1}) .* sigmoid_diff(z{i});
    end
    for i = 1:length(W)
        W{i} = W{i} - pre_factor * dz{i} * a{i}';
        b{i} = b{i} - pre_factor * sum(dz{i}, 2);
    end
end

function loss = compute_loss(Y, Y_prob)
    % Compute cross-entropy loss
    ln = log(Y_prob);
    loss = -mean(ln(Y == 1));
end

function plot_results(results)
    % Plot results
    figure;
    tiledlayout(2,1);
    nexttile;
    hold on;
    nexttile;
    hold on;
    for fn = fieldnames(results)'
        fn = fn{1};
        res = results.(fn);
        nexttile(1);
        plot(res.losses_train, 'DisplayName', fn);
        plot(res.losses_test, 'DisplayName', fn);
        nexttile(2);
        plot(res.accuracies_train, 'DisplayName', fn);
        plot(res.accuracies_test, 'DisplayName', fn);
    end
    nexttile(1);
    xlabel('Epoch');
    ylabel('CE Loss');
    legend('show');
    grid on;
    nexttile(2);
    xlabel('Epoch');
    ylabel('Accuracy (%)');
    legend('show');
    grid on;
end

function acc = accuracy(Y, Y_prob)
    % Returns accuracy of probability vectors against hot encoded values, as a value between 0 and 1
    correct = 0;
    for i = 1:size(Y, 2)
        [~, max_index_Y] = max(Y(:, i));
        [~, max_index_Y_prob] = max(Y_prob(:, i));
        if max_index_Y == max_index_Y_prob
            correct = correct + 1;
        end
    end
    acc = correct / size(Y, 2);
end

function [X_batch, Y_batch] = batch(X, Y, batch_size, i)
    % Returns the i'th batch of X and Y given the batch size
    j = (i-1)*batch_size + 1;
    batch_interval = [j, j + batch_size - 1];
    sample_size = size(X, 2);
    if batch_interval(2) > sample_size
        batch_interval(2) = sample_size;
    end
    X_batch = X(:, batch_interval(1):batch_interval(2));
    Y_batch = Y(:, batch_interval(1):batch_interval(2));
end

function Y_encoded = my_onehotencode(Y, min, max)
    % Returns a matrix with row wise stacking of one-hot-encoding of the contents of Y given 
    % a certain minimum value and maximum value to be encoded
    num_classes = max - min + 1;
    Y_encoded = zeros(num_classes, length(Y));
    for i = 1:length(Y)
        Y_encoded(Y(i) - min + 1, i) = 1;
    end
end

function s = sigmoid(z)
    % Sigmoid function
    s = 1 ./ (1 + exp(-z));
end

function diff = sigmoid_diff(z)
    % Derivative of the sigmoid function
    s = sigmoid(z);
    diff = s .* (1 - s);
end

function res = softmax(z)
    % Softmax function
    exp_z = exp(z);
    res = exp_z ./ sum(exp_z, 1);
end