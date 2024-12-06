clear all; close all; clc;

% Load the dataset
filePath = 'Workshop02/data/DATA_1.csv';
opts = detectImportOptions(filePath, 'Delimiter', ';', 'DecimalSeparator', ',');
data = readtable(filePath, opts);

% Feature selection
features = {'lpow', 'lspeed', 'dist', 'vol'};
target = 'Ra';

% Extract features and target
X = data{:, features};
y = data{:, target};

% Handle missing values
X = fillmissing(X, 'linear');
y = fillmissing(y, 'linear');

% Split data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv));
y_test = y(test(cv));

% Standardize the features
mu = mean(X_train);
sigma = std(X_train);
X_train_std = (X_train - mu) ./ sigma;
X_test_std = (X_test - mu) ./ sigma;

% Initialize results table
results = table();

% 1. Linear Regression
lin_model = fitlm(X_train_std, y_train);
y_pred = predict(lin_model, X_test_std);
results = [results; struct2table(struct( ...
    'Model', "Linear Regression", ...
    'MAE', mean(abs(y_test - y_pred)), ...
    'MSE', mean((y_test - y_pred).^2), ...
    'R2', lin_model.Rsquared.Ordinary))];

% 2. Random Forest
rng(42); % For reproducibility
rf_model = TreeBagger(100, X_train_std, y_train, 'Method', 'regression');
y_pred = predict(rf_model, X_test_std);
results = [results; struct2table(struct( ...
    'Model', "Random Forest", ...
    'MAE', mean(abs(y_test - y_pred)), ...
    'MSE', mean((y_test - y_pred).^2), ...
    'R2', 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2)))];

% Define the polynomial degree
degree = 2;

% Generate polynomial features
X_poly_train = poly_features(X_train_std, degree);
X_poly_test = poly_features(X_test_std, degree);

% Remove intercept column before ridge regression
X_poly_train_no_intercept = X_poly_train(:, 2:end);  % Remove first column (manual intercept)
X_poly_test_no_intercept = X_poly_test(:, 2:end);

% Perform Ridge Regression
lambda = 1;
ridge_model = ridge(y_train, X_poly_train_no_intercept, lambda, 0);  % Ridge regression

% Compute predictions
y_pred = [ones(size(X_poly_test_no_intercept, 1), 1), X_poly_test_no_intercept] * ridge_model;  % Add intercept for prediction

% Display results
disp('Model Comparison:');
disp(results);
% Plot residuals for each model
figure;
hold on;

% Linear Regression residuals
subplot(1, 3, 1);
y_pred_lin = predict(lin_model, X_test_std);
residuals_lin = y_test - y_pred_lin;
scatter(y_test, residuals_lin, 'filled');
yline(0, 'r--');
title('Linear Regression Residuals');
xlabel('Actual Values');
ylabel('Residuals');

% Random Forest residuals
subplot(1, 3, 2);
y_pred_rf = predict(rf_model, X_test_std);
residuals_rf = y_test - y_pred_rf;
scatter(y_test, residuals_rf, 'filled');
yline(0, 'r--');
title('Random Forest Residuals');
xlabel('Actual Values');
ylabel('Residuals');

% Polynomial Ridge Regression residuals
subplot(1, 3, 3);
residuals_poly = y_test - y_pred; % y_pred already computed for Ridge
scatter(y_test, residuals_poly, 'filled');
yline(0, 'r--');
title('Polynomial Ridge Regression Residuals');
xlabel('Actual Values');
ylabel('Residuals');

hold off;

% Utility function for generating polynomial features
function X_poly = poly_features(X, degree)
    X_poly = X;
    for d = 2:degree
        X_poly = [X_poly, X.^d];
    end
end
