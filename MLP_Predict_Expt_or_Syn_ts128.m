clear; close all; clc;

% Number of training runs to average results
numRuns = 3;

% Load full data file
dataFile = 'Expt676_Syn676_Combined_UnscoredData.txt';  % <-- Update as needed
dataTable = readtable(dataFile, 'Delimiter', '\t');

% Handle infinite values in column 30 (if needed)
if any(isinf(dataTable{:,30}))
    dataTable{isinf(dataTable{:,30}), 30} = mean(dataTable{~isinf(dataTable{:,30}), 30});
end

% Extract Type labels
types = string(dataTable{:, 4});  % Column 4 = 'Type' (Experiment or Synthetic)

% Separate data by type
expData = dataTable(strcmp(types, 'Experiment'), :);
synData = dataTable(strcmp(types, 'Synthetic'), :);

% Use first 128 of each for training, rest for testing
trainDataTable = [expData(1:128, :); synData(1:128, :)];
testDataTable = [expData(129:end, :); synData(129:end, :)];

% Extract features and labels
trainFeaturesRaw = trainDataTable{:, 10:56};  % Feature columns
testFeaturesRaw  = testDataTable{:, 10:56};
trainLabels = categorical(trainDataTable{:, 4}, {'Experiment', 'Synthetic'});
testLabels  = categorical(testDataTable{:, 4}, {'Experiment', 'Synthetic'});

% Preprocess features: fill missing and normalize
trainFeaturesRaw = fillmissing(trainFeaturesRaw, 'constant', mean(trainFeaturesRaw, 1, 'omitnan'));
testFeaturesRaw  = fillmissing(testFeaturesRaw, 'constant', mean(trainFeaturesRaw, 1, 'omitnan'));

mu = mean(trainFeaturesRaw, 1);
sigma = std(trainFeaturesRaw, 0, 1);
trainFeatures = (trainFeaturesRaw - mu) ./ sigma;
testFeatures  = (testFeaturesRaw - mu) ./ sigma;

% Define MLP architecture
layers = [
    featureInputLayer(size(trainFeatures, 2), 'Normalization', 'none')
    fullyConnectedLayer(128, 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(64, 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(2)  % 2 output classes
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {testFeatures, testLabels}, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'none');

% Store accuracy and predictions from each run
allAccuracies = zeros(numRuns, 1);
allPredictedLabels = [];

for run = 1:numRuns
    % Train network
    net = trainNetwork(trainFeatures, trainLabels, layers, options);
    
    % Predict
    predictedLabels = classify(net, testFeatures);
    
    % Ensure categories are aligned before comparison
    predictedLabels = categorical(predictedLabels, {'Experiment', 'Synthetic'});
    
    % Accuracy calculation
    allAccuracies(run) = mean(predictedLabels == testLabels) * 100;

    % Save predictions from the final run
    if run == numRuns
        allPredictedLabels = predictedLabels;
    end
end

% Final average accuracy
meanAccuracy = mean(allAccuracies);
fprintf('Mean Test Accuracy over %d run(s): %.2f%%\n', numRuns, meanAccuracy);

% Confusion matrix normalized as percentages per actual class (row-normalized)
figure;
cm = confusionchart(testLabels, allPredictedLabels, 'Normalization', 'row-normalized');
cm.Title = sprintf('Confusion Matrix (Mean Accuracy: %.2f%%)', meanAccuracy);
cm.RowSummary = 'row-normalized';  % Show row percentages on the side
cm.ColumnSummary = 'column-normalized'; % Optional: Show col percentages on bottom
