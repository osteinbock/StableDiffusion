clear; close all; clc;

numRuns = 3;  % Number of runs

% Load data
dataFile = 'Expt676_Syn676_Combined_UnscoredData.txt';  % Replace with actual file if needed
dataTable = readtable(dataFile, 'Delimiter', '\t');

% Handle infinite values (column 30)
if any(isinf(dataTable{:,30}))
    dataTable{isinf(dataTable{:,30}), 30} = mean(dataTable{~isinf(dataTable{:,30}), 30});
end

% Extract 'Type' labels
types = string(dataTable{:, 4});  % Column 4 = 'Type' (Experiment or Synthetic)

% Separate Experiment and Synthetic
expData = dataTable(strcmp(types, 'Experiment'), :);
synData = dataTable(strcmp(types, 'Synthetic'), :);

% Split into training (128 each) and testing (rest)
trainDataTable = [expData(1:128, :); synData(1:128, :)];
testDataTable  = [expData(129:end, :); synData(129:end, :)];

% Extract features (columns 10 to 56) and labels
X_train = trainDataTable{:, 10:56};
X_test  = testDataTable{:, 10:56};
Y_train = categorical(trainDataTable{:, 4}, {'Experiment', 'Synthetic'});
Y_test  = categorical(testDataTable{:, 4}, {'Experiment', 'Synthetic'});

% Fill missing values in features
X_train = fillmissing(X_train, 'constant', mean(X_train, 1, 'omitnan'));
X_test  = fillmissing(X_test, 'constant', mean(X_train, 1, 'omitnan'));

% Normalize features using training stats
mu = mean(X_train, 1);
sigma = std(X_train, 0, 1);
X_train = (X_train - mu) ./ sigma;
X_test  = (X_test - mu) ./ sigma;

numTrees = 100;
allAccuracies = zeros(numRuns, 1);

for run = 1:numRuns
    rng(run);  % Set seed for reproducibility per run
    
    % Train Random Forest (TreeBagger)
    rfModel = TreeBagger(numTrees, X_train, Y_train, ...
        'OOBPrediction', 'On', ...
        'Method', 'classification');
    
    % Predict on test data
    Y_pred = predict(rfModel, X_test);
    Y_pred = categorical(Y_pred, {'Experiment', 'Synthetic'});
    
    % Compute accuracy
    allAccuracies(run) = mean(Y_pred == Y_test) * 100;
    
    % Save predictions for last run to show confusion matrix
    if run == numRuns
        finalPredictedLabels = Y_pred;
        finalModel = rfModel;
    end
end

meanAccuracy = mean(allAccuracies);
fprintf('Random Forest Mean Test Accuracy over %d runs: %.2f%%\n', numRuns, meanAccuracy);

% Confusion Matrix normalized by rows (actual classes) for last run
figure;
cm = confusionchart(Y_test, finalPredictedLabels, 'Normalization', 'row-normalized');
cm.Title = sprintf('Random Forest Confusion Matrix (Accuracy: %.2f%%)', allAccuracies(end));
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Optional: Out-of-Bag Error Plot for last run
figure;
oobErrorBaggedEnsemble = oobError(finalModel);
plot(oobErrorBaggedEnsemble);
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');
title('Out-of-Bag Error vs. Number of Trees');
grid on;
