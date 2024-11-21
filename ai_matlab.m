% Load Iris dataset
load fisheriris;
X = meas;
Y = species;

% Standardize features
X_scaled = (X - mean(X)) ./ std(X);

% Convert labels to numeric
[unique_labels, ~, Y_numeric] = unique(Y);

% One-hot encode labels manually
Y_encoded = zeros(length(Y_numeric), length(unique(Y_numeric)));
for i = 1:length(Y_numeric)
    Y_encoded(i, Y_numeric(i)) = 1;
end

% Split data
cv = cvpartition(Y_numeric, 'HoldOut', 0.2);
X_train = X_scaled(training(cv), :);
Y_train = Y_encoded(training(cv), :);
X_test = X_scaled(test(cv), :);
Y_test = Y_encoded(test(cv), :);

% Create neural network with default activation functions
net = patternnet([10 8 6]);
net.trainParam.epochs = 200;
net.trainParam.lr = 0.01;

% Train network
[net, tr] = train(net, X_train', Y_train');

% Evaluate
Y_pred_test = net(X_test');
[~, predicted_classes] = max(Y_pred_test, [], 1);
[~, true_classes] = max(Y_test', [], 1);
accuracy = mean(predicted_classes == true_classes) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

% Prediction function
function predicted_class = predict_iris(net, features, unique_labels)
    % Scale input
    features_scaled = (features - mean(features)) ./ std(features);
    
    % Predict
    prediction = net(features_scaled');
    [~, predicted_idx] = max(prediction);
    predicted_class = unique_labels{predicted_idx};
end

% Example usage
sample_features = [5.1, 3.5, 1.4, 0.2];
predicted_class = predict_iris(net, sample_features, unique_labels);
disp(predicted_class);