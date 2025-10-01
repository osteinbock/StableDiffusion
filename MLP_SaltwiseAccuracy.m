clear all;
clc;
close all;

file_prefixes = [1, 2, 4, 6, 8, 16, 32, 64, 128];
numSizes = length(file_prefixes);
numRuns = 3;

% Store mean accuracy for each training size and mode
% Columns: 1 = Real, 2 = Synthetic, 3 = Combined
allAccuracies = zeros(numSizes, 3);
syntheticStd = zeros(numSizes, 1);
combinedStd = zeros(numSizes, 1);
realStd = zeros(numSizes, 1);

% Salt-wise accuracy: now include ALL training sizes
numSalts = 7;
saltAccuracyMatrix = nan(numSalts, numSizes, 3);  % salts × prefixes × dataTypes

% Load test data
testFile = 'Cropped_ExptFile_ExcludingFirst128PerSalt.txt';
testTable = readtable(testFile, 'Delimiter', '\t');
testTable{isinf(testTable{:,30}), 30} = mean(testTable{~isinf(testTable{:,30}), 30});
testSalt = testTable{:, 3};
testMetricsOrig = testTable{:, 9:55};
testCategoriesOrig = string(testSalt);

uniqueSaltNames = string(unique(testSalt, 'stable'));
saltNames = cellstr(uniqueSaltNames);  % Initialize saltNames once here

for i = 1:numSizes
    prefix = file_prefixes(i);
    accuraciesReal = [];
    accuraciesSynth = [];
    accuraciesCombined = [];

    if prefix <= 8
        reps = 1:3;
    else
        reps = 1;
    end

    saltAccRealRuns = zeros(numSalts, length(reps));
    saltAccSynthRuns = zeros(numSalts, length(reps));
    saltAccCombRuns  = zeros(numSalts, length(reps));

    repCounter = 1;
    for rep = reps
        realFile = sprintf('exptData%dImages_%d_Unscored.txt', prefix, rep);
        trainTableReal = readtable(realFile, 'Delimiter', '\t');

        synthFile = sprintf('ts%d_%d_Data_unscored.txt', prefix, rep);
        trainTableSynth = readtable(synthFile, 'Delimiter', '\t');

        % --- REAL ---
        runAccuracies = zeros(numRuns,1);
        [trainData, trainLabels, testData, testLabels] = preprocess(trainTableReal, testMetricsOrig, testCategoriesOrig);
        for run = 1:numRuns
            layers = getMLPLayers(size(trainData, 2), numel(categories(trainLabels)));
            options = getTrainingOptions(testData, testLabels);
            net = trainNetwork(trainData, trainLabels, layers, options);
            predicted = classify(net, testData);
            runAccuracies(run) = mean(predicted == testLabels) * 100;

            % Salt-wise
            thisRunSaltAcc = zeros(numSalts,1);
            uniqueSalts = categories(testLabels);
            for s = 1:numSalts
                idx = testLabels == uniqueSalts{s};
                if any(idx)
                    thisRunSaltAcc(s) = mean(predicted(idx) == testLabels(idx)) * 100;
                else
                    thisRunSaltAcc(s) = NaN;
                end
            end
            saltAccRealRuns(:, repCounter) = thisRunSaltAcc;
        end
        accuraciesReal(end+1) = mean(runAccuracies);

        % --- SYNTHETIC ---
        runAccuracies = zeros(numRuns,1);
        [trainData, trainLabels, testData, testLabels] = preprocess(trainTableSynth, testMetricsOrig, testCategoriesOrig);
        for run = 1:numRuns
            layers = getMLPLayers(size(trainData, 2), numel(categories(trainLabels)));
            options = getTrainingOptions(testData, testLabels);
            net = trainNetwork(trainData, trainLabels, layers, options);
            predicted = classify(net, testData);
            runAccuracies(run) = mean(predicted == testLabels) * 100;

            % Salt-wise
            thisRunSaltAcc = zeros(numSalts,1);
            uniqueSalts = categories(testLabels);
            for s = 1:numSalts
                idx = testLabels == uniqueSalts{s};
                if any(idx)
                    thisRunSaltAcc(s) = mean(predicted(idx) == testLabels(idx)) * 100;
                else
                    thisRunSaltAcc(s) = NaN;
                end
            end
            saltAccSynthRuns(:, repCounter) = thisRunSaltAcc;
        end
        accuraciesSynth(end+1) = mean(runAccuracies);

        % --- COMBINED ---
        trainTableCombined = [trainTableReal; trainTableSynth];
        runAccuracies = zeros(numRuns,1);
        [trainData, trainLabels, testData, testLabels] = preprocess(trainTableCombined, testMetricsOrig, testCategoriesOrig);
        for run = 1:numRuns
            layers = getMLPLayers(size(trainData, 2), numel(categories(trainLabels)));
            options = getTrainingOptions(testData, testLabels);
            net = trainNetwork(trainData, trainLabels, layers, options);
            predicted = classify(net, testData);
            runAccuracies(run) = mean(predicted == testLabels) * 100;

            % Salt-wise
            thisRunSaltAcc = zeros(numSalts,1);
            uniqueSalts = categories(testLabels);
            for s = 1:numSalts
                idx = testLabels == uniqueSalts{s};
                if any(idx)
                    thisRunSaltAcc(s) = mean(predicted(idx) == testLabels(idx)) * 100;
                else
                    thisRunSaltAcc(s) = NaN;
                end
            end
            saltAccCombRuns(:, repCounter) = thisRunSaltAcc;
        end
        accuraciesCombined(end+1) = mean(runAccuracies);

        repCounter = repCounter + 1;
    end

    % Store means and stds
    allAccuracies(i, 1) = mean(accuraciesReal);
    allAccuracies(i, 2) = mean(accuraciesSynth);
    allAccuracies(i, 3) = mean(accuraciesCombined);
    realStd(i) = std(accuraciesReal);
    syntheticStd(i) = std(accuraciesSynth);
    combinedStd(i) = std(accuraciesCombined);

    % Store salt-wise accuracy for ALL prefixes (no more filtering)
    saltAccuracyMatrix(:, i, 1) = mean(saltAccRealRuns, 2, 'omitnan');
    saltAccuracyMatrix(:, i, 2) = mean(saltAccSynthRuns, 2, 'omitnan');
    saltAccuracyMatrix(:, i, 3) = mean(saltAccCombRuns, 2, 'omitnan');

    fprintf('Prefix %3d | Real: %.2f%% | Synth: %.2f%% | Comb: %.2f%%\n', ...
        prefix, allAccuracies(i,1), allAccuracies(i,2), allAccuracies(i,3));
end

% --- PLOT ACCURACIES ---
figure;
logx = log2(file_prefixes);

errorbar(logx, allAccuracies(:,1), realStd, '-o', 'LineWidth', 2, 'Color', [0.4 0.8 0.4], 'MarkerSize', 8); hold on;
errorbar(logx, allAccuracies(:,2), syntheticStd, '-s', 'LineWidth', 2, 'Color', [1 0.4 0.4], 'MarkerSize', 8);
errorbar(logx, allAccuracies(:,3), combinedStd, '-^', 'LineWidth', 2, 'Color', [0.2 0.6 1], 'MarkerSize', 8);

grid on;
xlabel('Number of Training Images');
ylabel('Test Accuracy (%)');
legend({'Real Data', 'Synthetic Data', 'Combined'}, 'Location', 'southeast');
xticks(logx);
xticklabels(string(file_prefixes));
set(gca, 'FontSize', 12);
set(gcf, 'color', 'w');

% --- SAVE RESULTS ---
datasetTypes = {'Real', 'Synthetic', 'Combined'};
save('MLP_Accuracy_Results.mat', 'allAccuracies', 'realStd', 'syntheticStd', 'combinedStd', 'file_prefixes');
save('MLP_Saltwise_Accuracy.mat', 'saltAccuracyMatrix', 'saltNames', 'file_prefixes', 'datasetTypes');

%% 
%% ---------------------- Local Functions ----------------------

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
    
    trainFilled = fillmissing(trainMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));
    testFilled = fillmissing(testMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));
    trainMean = mean(trainFilled, 1);
    trainStd = std(trainFilled, 0, 1);
    trainData = (trainFilled - trainMean) ./ trainStd;
    testData = (testFilled - trainMean) ./ trainStd;
end

function layers = getMLPLayers(inputSize, numClasses)
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'none')
        fullyConnectedLayer(1024, 'WeightsInitializer', 'he')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(512, 'WeightsInitializer', 'he')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(256, 'WeightsInitializer', 'he')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(128, 'WeightsInitializer', 'he')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
end

function options = getTrainingOptions(testData, testLabels)
    options = trainingOptions('adam', ...
        'MiniBatchSize', 128, ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 0.01, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {testData, testLabels}, ...
        'ValidationPatience', 5, ...
        'Verbose', false, ...
        'Plots', 'none');
end
