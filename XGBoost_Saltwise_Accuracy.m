clear all;
close all;

file_prefixes = [1, 2, 4, 6, 8, 16, 32, 64, 128];
numSizes = length(file_prefixes);
numRuns = 3;
numSalts = 7;

% Store overall accuracy stats
allAccuracies = zeros(numSizes, 3);  % Columns: [Real, Synthetic, Combined]
realStd = zeros(numSizes, 1);
syntheticStd = zeros(numSizes, 1);
combinedStd = zeros(numSizes, 1);

% Salt-wise accuracy for ALL sizes
saltAccuracyMatrix = nan(numSalts, numSizes, 3);  % salts × prefixes × [Real, Synth, Comb]
saltNames = [];

% Load test data
testFile = 'Cropped_ExptFile_ExcludingFirst128PerSalt.txt';
testTable = readtable(testFile, 'Delimiter', '\t');
testTable{isinf(testTable{:,30}), 30} = mean(testTable{~isinf(testTable{:,30}), 30});
testSalt = testTable{:, 3};
testMetricsOrig = testTable{:, 9:55};
testCategoriesOrig = string(testSalt);
uniqueSaltNames = string(unique(testSalt, 'stable'));

% Python environment
pyenv;
try
    xgb = py.importlib.import_module('xgboost');
    np = py.importlib.import_module('numpy');
    fprintf('XGBoost and NumPy successfully imported.\n');
catch
    error('Python modules not found. Please install xgboost and numpy.');
end

% --- Main Loop ---
for i = 1:numSizes
    prefix = file_prefixes(i);
    if prefix <= 8
        reps = 1:3;
    else
        reps = 1;
    end
    accuraciesReal = [];
    accuraciesSynth = [];
    accuraciesCombined = [];

    saltAccRealRuns = zeros(numSalts, length(reps));
    saltAccSynthRuns = zeros(numSalts, length(reps));
    saltAccCombRuns  = zeros(numSalts, length(reps));

    repCounter = 1;

    for rep = reps
        realFile = sprintf('exptData%dImages_%d_Unscored.txt', prefix, rep);
        synthFile = sprintf('ts%d_%d_Data_unscored.txt', prefix, rep);

        if ~isfile(realFile) || ~isfile(synthFile)
            warning('Missing file(s) for prefix %d rep %d. Skipping.', prefix, rep);
            continue;
        end

        trainTableReal = readtable(realFile, 'Delimiter', '\t');
        trainTableSynth = readtable(synthFile, 'Delimiter', '\t');
        trainTableCombined = [trainTableReal; trainTableSynth];

        % Real
        [accReal, saltReal] = runXGBoost(trainTableReal, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns, uniqueSaltNames);
        accuraciesReal(end+1) = mean(accReal);
        saltAccRealRuns(:, repCounter) = mean(saltReal, 2, 'omitnan');

        % Synthetic
        [accSynth, saltSynth] = runXGBoost(trainTableSynth, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns, uniqueSaltNames);
        accuraciesSynth(end+1) = mean(accSynth);
        saltAccSynthRuns(:, repCounter) = mean(saltSynth, 2, 'omitnan');

        % Combined
        [accCombined, saltComb] = runXGBoost(trainTableCombined, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns, uniqueSaltNames);
        accuraciesCombined(end+1) = mean(accCombined);
        saltAccCombRuns(:, repCounter) = mean(saltComb, 2, 'omitnan');

        repCounter = repCounter + 1;
    end

    % Store overall results
    allAccuracies(i, :) = [mean(accuraciesReal), mean(accuraciesSynth), mean(accuraciesCombined)];
    realStd(i) = std(accuraciesReal);
    syntheticStd(i) = std(accuraciesSynth);
    combinedStd(i) = std(accuraciesCombined);

    % Store salt-wise accuracy for ALL training sizes
    saltAccuracyMatrix(:, i, 1) = mean(saltAccRealRuns, 2, 'omitnan');
    saltAccuracyMatrix(:, i, 2) = mean(saltAccSynthRuns, 2, 'omitnan');
    saltAccuracyMatrix(:, i, 3) = mean(saltAccCombRuns, 2, 'omitnan');

    if isempty(saltNames)
        saltNames = cellstr(uniqueSaltNames);
    end

    fprintf('Prefix %3d | Real: %.2f%% | Synth: %.2f%% | Comb: %.2f%%\n', ...
        prefix, allAccuracies(i,1), allAccuracies(i,2), allAccuracies(i,3));
end

% --- Plot Results ---
figure;
logx = log2(file_prefixes);

errorbar(logx, allAccuracies(:,1), realStd, '-o', 'LineWidth', 2, 'Color', [0.2 0.6 1], 'MarkerSize', 8); hold on;
errorbar(logx, allAccuracies(:,2), syntheticStd, '-s', 'LineWidth', 2, 'Color', [1 0.4 0.4], 'MarkerSize', 8);
errorbar(logx, allAccuracies(:,3), combinedStd, '-^', 'LineWidth', 2, 'Color', [0.4 0.8 0.4], 'MarkerSize', 8);

grid on;
xlabel('log_2(Number of Training Images)');
ylabel('Test Accuracy (%)');
title('XGBoost Accuracy: Real vs Synthetic vs Combined');
legend({'Real Data', 'Synthetic Data', 'Combined'}, 'Location', 'southeast');
xticks(logx);
xticklabels(string(file_prefixes));
set(gca, 'FontSize', 12);
set(gcf, 'color', 'w');

% --- Save Results ---
datasetTypes = {'Real', 'Synthetic', 'Combined'};
%save('XGB_Accuracy_Results.mat', 'file_prefixes', 'allAccuracies', 'realStd', 'syntheticStd', 'combinedStd');
save('XGB_Saltwise_Accuracy.mat', 'saltAccuracyMatrix', 'saltNames', 'file_prefixes', 'datasetTypes');



%% ----------- Local function for running XGBoost -----------

function [accuracies, saltAccMatrix] = runXGBoost(trainTable, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns, uniqueSaltNames)
    accuracies = zeros(numRuns, 1);
    numSalts = length(uniqueSaltNames);
    saltAccMatrix = nan(numSalts, numRuns);

    [trainData, trainLabels, testData, testLabels] = preprocess(trainTable, testMetricsOrig, testCategoriesOrig);

    [labelCats, ~, trainLabelsNum] = unique(trainLabels);
    [~, testLabelsNum] = ismember(testLabels, labelCats);
    trainLabelsNum = trainLabelsNum - 1;
    testLabelsNum = testLabelsNum - 1;

    dtrain = xgb.DMatrix(py.numpy.array(trainData), pyargs('label', py.numpy.array(trainLabelsNum)));
    dtest = xgb.DMatrix(py.numpy.array(testData), pyargs('label', py.numpy.array(testLabelsNum)));

    params = py.dict(pyargs( ...
        'objective', 'multi:softmax', ...
        'num_class', int32(numel(labelCats)), ...
        'max_depth', int32(6), ...
        'eta', 0.3, ...
        'eval_metric', 'mlogloss'));

    num_round = int32(100);

    for run = 1:numRuns
        params{'seed'} = int32(run);
        model = xgb.train(params, dtrain, num_round);
        preds = double(model.predict(dtest));
        predictedLabels = categorical(cellstr(labelCats(preds + 1)));

        accuracies(run) = sum(predictedLabels == testLabels) / numel(testLabels) * 100;

        for s = 1:numSalts
            idx = testCategoriesOrig == uniqueSaltNames{s};
            if any(idx)
                saltAccMatrix(s, run) = mean(predictedLabels(idx) == testLabels(idx)) * 100;
            else
                saltAccMatrix(s, run) = NaN;
            end
        end
    end
end



%% ----------- Local function for preprocessing data -----------

function [trainData, trainLabels, testData, testLabels] = preprocess(trainTable, testMetricsOrig, testCategoriesOrig)
    trainSalt = trainTable{:, 3};
    trainMetrics = trainTable{:, 9:55};
    trainCategories = string(trainSalt);

    [uniqueCategories, ~, trainLabelsRaw] = unique(trainCategories);
    [~, locb] = ismember(testCategoriesOrig, uniqueCategories);
    validIdx = locb > 0;
    testLabelsRaw = locb(validIdx);
    testMetrics = testMetricsOrig(validIdx, :);

    trainLabels = categorical(trainLabelsRaw);
    testLabels = categorical(testLabelsRaw);

    trainMetrics(~isfinite(trainMetrics)) = NaN;
    testMetrics(~isfinite(testMetrics)) = NaN;

    colMean = mean(trainMetrics, 1, 'omitnan');
    trainFilled = fillmissing(trainMetrics, 'constant', colMean);
    testFilled = fillmissing(testMetrics, 'constant', colMean);

    trainMean = mean(trainFilled, 1);
    trainStd = std(trainFilled, 0, 1);
    trainStd(trainStd == 0) = 1;

    trainData = (trainFilled - trainMean) ./ trainStd;
    testData = (testFilled - trainMean) ./ trainStd;

    trainData(~isfinite(trainData)) = 0;
    testData(~isfinite(testData)) = 0;
end
