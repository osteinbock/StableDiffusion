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
testDataTable  = [expData(129:end, :); synData(129:end, :)];

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

% Setup Python environment for XGBoost
pyenv;
try
    xgb = py.importlib.import_module('xgboost');
    np = py.importlib.import_module('numpy');
    fprintf('XGBoost and NumPy successfully imported.\n');
catch
    error('Python modules not found. Please install xgboost and numpy.');
end

% Convert MATLAB categorical labels to numeric (0 and 1)
[labelCats, ~, trainLabelsNum] = unique(trainLabels);
[~, testLabelsNum] = ismember(testLabels, labelCats);
trainLabelsNum = trainLabelsNum - 1;  % XGBoost expects zero-based labels
testLabelsNum = testLabelsNum - 1;

% Convert data to Python numpy arrays
dtrain = xgb.DMatrix(py.numpy.array(trainFeatures), pyargs('label', py.numpy.array(trainLabelsNum)));
dtest  = xgb.DMatrix(py.numpy.array(testFeatures), pyargs('label', py.numpy.array(testLabelsNum)));

% Define XGBoost parameters
params = py.dict(pyargs( ...
    'objective', 'multi:softmax', ...
    'num_class', int32(numel(labelCats)), ...
    'max_depth', int32(6), ...
    'eta', 0.3, ...
    'eval_metric', 'mlogloss'));

num_round = int32(100);

allAccuracies = zeros(numRuns,1);
allPredictedLabels = [];

for run = 1:numRuns
    params{'seed'} = int32(run);
    model = xgb.train(params, dtrain, num_round);
    preds = double(model.predict(dtest));
    
    % Convert predictions back to categorical labels
    predictedLabels = categorical(cellstr(labelCats(preds + 1)), {'Experiment', 'Synthetic'});
    
    % Accuracy calculation
    allAccuracies(run) = mean(predictedLabels == testLabels) * 100;
    
    % Save predictions from final run
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
cm.RowSummary = 'row-normalized';   % Show row percentages on side
cm.ColumnSummary = 'column-normalized'; % Optional: col percentages on bottom
