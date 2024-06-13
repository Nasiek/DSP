clear;
close all;
clc;

% Path to the FASHION MNIST dataset
datasetPath = 'fashion-mnist';

% Check if the dataset is already downloaded
if ~exist(datasetPath, 'dir')
    mkdir(datasetPath);
    
    % Download the dataset
    url = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/';
    files = {'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', ...
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'};
    
    for i = 1:numel(files)
        websave(fullfile(datasetPath, files{i}), [url files{i}]);
    end
end

% Unzip the files
gunzip(fullfile(datasetPath, 'train-images-idx3-ubyte.gz'));
gunzip(fullfile(datasetPath, 'train-labels-idx1-ubyte.gz'));
gunzip(fullfile(datasetPath, 't10k-images-idx3-ubyte.gz'));
gunzip(fullfile(datasetPath, 't10k-labels-idx1-ubyte.gz'));

% Load the dataset
[trainImages, trainLabels] = loadMNISTImages(fullfile(datasetPath, 'train-images-idx3-ubyte'), ...
                                             fullfile(datasetPath, 'train-labels-idx1-ubyte'));
[testImages, testLabels] = loadMNISTImages(fullfile(datasetPath, 't10k-images-idx3-ubyte'), ...
                                           fullfile(datasetPath, 't10k-labels-idx1-ubyte'));

trainImages = double(trainImages) / 255;
testImages = double(testImages) / 255;

% Reshape Fashion MNIST data to match the structure
training.images = permute(trainImages, [2, 3, 1]); % Reshape to 28x28x60000
training.labels = trainLabels;
training.count = size(trainImages, 4);
training.width = size(trainImages, 2);
training.height = size(trainImages, 1);

test.images = permute(testImages, [2, 3, 1]); % Reshape to 28x28x10000
test.labels = testLabels;
test.count = size(testImages, 4);
test.width = size(testImages, 2);
test.height = size(testImages, 1);

clearvars -except training test;

% Function to load MNIST images
function [images, labels] = loadMNISTImages(imageFile, labelFile)
    % Open image file
    fid = fopen(imageFile, 'r', 'b');
    assert(fid ~= -1, ['Could not open ', imageFile, '']);
    
    % Read file identifier 
    fileIdentifier = fread(fid, 1, 'int32', 0, 'ieee-be');
    assert(fileIdentifier == 2051, ['Invalid file identifier number in ', imageFile, '']);
    
    % Read dimensions
    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fid, 1, 'int32', 0, 'ieee-be');
    
    % Read image data
    images = fread(fid, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images, [2, 1, 3]);
    fclose(fid);
    
    % Open label file
    fid = fopen(labelFile, 'r', 'b');
    assert(fid ~= -1, ['Could not open ', labelFile, '']);
    
    % Read file identifier
    fileIdentifier = fread(fid, 1, 'int32', 0, 'ieee-be');
    assert(fileIdentifier == 2049, ['Invalid file identifier in ', labelFile, '']);
    
    % Read label data
    numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels = fread(fid, inf, 'unsigned char');
    fclose(fid);
    
    assert(numImages == numLabels, 'Number of images does not match number of labels');
    
    % Reshape images to 4D array (number of images, width, height, channels)
    images = reshape(images, numRows, numCols, 1, numImages);
    images = permute(images, [4, 2, 1, 3]);
end
