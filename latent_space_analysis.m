% Load Fashion MNIST data using Fashion_MNIST.m script
Fashion_MNIST;

% Extract train and test data from loaded structures
X_train = training.images;
X_test = test.images;

% Reshape data to be compatible with the network
X_train = reshape(X_train, [28, 28, 1, size(X_train, 3)]);
X_test = reshape(X_test, [28, 28, 1, size(X_test, 3)]);

% Flatten images for PCA
X_train_flat = reshape(X_train, [], size(X_train, 4))';
X_test_flat = reshape(X_test, [], size(X_test, 4))';

% Perform PCA for dimensionality reduction
num_components = 10;  % Number of components to keep
[coeff, score, ~, ~, explained] = pca(X_train_flat);
X_train_pca = score(:, 1:num_components);
X_test_pca = (X_test_flat - mean(X_test_flat)) * coeff(:, 1:num_components);

% Reconstruct using PCA
X_test_reconstructed_flat = X_test_pca * coeff(:, 1:num_components)' + mean(X_train_flat);

% Reshape back to image dimensions for visualization
X_test_reconstructed = reshape(X_test_reconstructed_flat', size(X_test));

% Display original and reconstructed images
num_images = 10;
figure;
tiledlayout(2, num_images);
for i = 1:num_images
    % Select a random test image
    idx = randi(size(X_test, 4));
    
    % Plot original and reconstructed images side by side
    nexttile;
    imshow(squeeze(X_test(:, :, 1, idx)));
    title('Original');
    
    nexttile;
    imshow(squeeze(X_test_reconstructed(:, :, 1, idx)), []);
    title('Reconstructed');
end

% Plot explained variance
figure;
plot(cumsum(explained), 'o-');
xlabel('Number of Components');
ylabel('Cumulative Explained Variance (%)');
title('Explained Variance by PCA Components');

% Perform clustering on PCA-transformed data
num_clusters = 10; % Estimate number of clusters
silhouette_scores = zeros(1, 15);
for k = 2:15
    [cluster_labels, ~] = kmeans(X_train_pca, k, 'MaxIter', 1000, 'Replicates', 10);
    silhouette_scores(k) = mean(silhouette(X_train_pca, cluster_labels));
end

% Plot silhouette scores to determine optimal number of clusters
figure;
plot(2:15, silhouette_scores(2:15), 'o-');
xlabel('Number of Clusters');
ylabel('Silhouette Score');
title('Silhouette Scores for Different Numbers of Clusters');

% Determine optimal number of clusters
[~, optimal_clusters] = max(silhouette_scores);
fprintf('Optimal number of clusters: %d\n', optimal_clusters);

% Perform clustering with the optimal number of clusters
[optimal_cluster_labels, ~] = kmeans(X_train_pca, optimal_clusters, 'MaxIter', 1000, 'Replicates', 10);

% Visualize clusters in 2D PCA space
figure;
scatter(X_train_pca(:, 1), X_train_pca(:, 2), 10, optimal_cluster_labels, 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('Clusters Visualized in PCA Space');
colormap(jet(optimal_clusters));
colorbar;
