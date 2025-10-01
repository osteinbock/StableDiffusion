clear all;
close all;

file_prefixes = [1, 2, 4, 6, 8, 16, 32, 64, 128];
numSizes = length(file_prefixes);
numRuns = 3;
numTrees = 100;

% Store mean accuracy for each training size and mode
% Columns: 1 = Real, 2 = Synthetic, 3 = Combined
allAccuracies = zeros(numSizes, 3);
realStd = zeros(numSizes, 1);
syntheticStd = zeros(numSizes, 1);
combinedStd = zeros(numSizes, 1);

% --- Salt-wise accuracy tracking ---
numSalts = 7;  % Known fixed number of salt classes
saltAccuracyMatrix = nan(numSalts, numSizes, 3);  % salts × training sizes × dataTypes
saltNames = [];

% Load fixed test data
testFile = 'Cropped_ExptFile_ExcludingFirst128PerSalt.txt';
testTable = readtable(testFile, 'Delimiter', '\t');
testTable{isinf(testTable{:,30}), 30} = mean(testTable{~isinf(testTable{:,30}), 30});
testSalt = testTable{:, 3};
testMetricsOrig = testTable{:, 9:55};
testCategoriesOrig = string(testSalt);
uniqueSaltNames = string(unique(testSalt, 'stable'));

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

    % Initialize salt-wise accuracy for this training size
    saltAccRealRuns = zeros(numSalts, length(reps));
    saltAccSynthRuns = zeros(numSalts, length(reps));
    saltAccCombRuns  = zeros(numSalts, length(reps));
    repCounter = 1;

    for rep = reps
        % --- Load training sets ---
        realFile = sprintf('exptData%dImages_%d_Unscored.txt', prefix, rep);
        trainTableReal = readtable(realFile, 'Delimiter', '\t');

        synthFile = sprintf('ts%d_%d_Data_unscored.txt', prefix, rep);
        trainTableSynth = readtable(synthFile, 'Delimiter', '\t');

        % --- REAL ---
        for run = 1:numRuns
            [trainData, trainLabels, testData, testLabels] = preprocessRF(trainTableReal, testMetricsOrig, testCategoriesOrig);
            rfModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'classification', 'OOBPrediction', 'On');
            predicted = predict(rfModel, testData);
            predicted = categorical(predicted);
            acc = mean(predicted == testLabels) * 100;
            accuraciesReal(end+1) = acc;

            % Salt-wise
            thisRunSaltAcc = zeros(numSalts, 1);
            uniqueSalts = categories(testLabels);
            if isempty(saltNames)
                saltNames = cellstr(uniqueSaltNames);
            end
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

        % --- SYNTHETIC ---
        for run = 1:numRuns
            [trainData, trainLabels, testData, testLabels] = preprocessRF(trainTableSynth, testMetricsOrig, testCategoriesOrig);
            rfModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'classification', 'OOBPrediction', 'On');
            predicted = predict(rfModel, testData);
            predicted = categorical(predicted);
            acc = mean(predicted == testLabels) * 100;
            accuraciesSynth(end+1) = acc;

            % Salt-wise
            thisRunSaltAcc = zeros(numSalts, 1);
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

        % --- COMBINED ---
        trainTableCombined = [trainTableReal; trainTableSynth];
        for run = 1:numRuns
            [trainData, trainLabels, testData, testLabels] = preprocessRF(trainTableCombined, testMetricsOrig, testCategoriesOrig);
            rfModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'classification', 'OOBPrediction', 'On');
            predicted = predict(rfModel, testData);
            predicted = categorical(predicted);
            acc = mean(predicted == testLabels) * 100;
            accuraciesCombined(end+1) = acc;

            % Salt-wise
            thisRunSaltAcc = zeros(numSalts, 1);
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

        repCounter = repCounter + 1;
    end

    % Store mean accuracies
    allAccuracies(i, 1) = mean(accuraciesReal);
    allAccuracies(i, 2) = mean(accuraciesSynth);
    allAccuracies(i, 3) = mean(accuraciesCombined);
    realStd(i) = std(accuraciesReal);
    syntheticStd(i) = std(accuraciesSynth);
    combinedStd(i) = std(accuraciesCombined);

    % Store salt-wise accuracy for ALL prefixes
    saltAccuracyMatrix(:, i, 1) = mean(saltAccRealRuns, 2, 'omitnan');
    saltAccuracyMatrix(:, i, 2) = mean(saltAccSynthRuns, 2, 'omitnan');
    saltAccuracyMatrix(:, i, 3) = mean(saltAccCombRuns, 2, 'omitnan');

    fprintf('Prefix %3d | Real: %.2f%% | Synth: %.2f%% | Comb: %.2f%%\n', ...
        prefix, allAccuracies(i,1), allAccuracies(i,2), allAccuracies(i,3));
end

% --- Plot Accuracy with Error Bars ---
figure;
logx = log2(file_prefixes);

errorbar(logx, allAccuracies(:,1), realStd, '-o', 'LineWidth', 2, 'Color', [0.4 0.8 0.4], 'MarkerSize', 8); hold on;
errorbar(logx, allAccuracies(:,2), syntheticStd, '-s', 'LineWidth', 2, 'Color', [1 0.4 0.4], 'MarkerSize', 8);
errorbar(logx, allAccuracies(:,3), combinedStd, '-^', 'LineWidth', 2, 'Color', [0.2 0.6 1], 'MarkerSize', 8);

grid on;
xlabel('Number of Training Images');
ylabel('Test Accuracy (%)');
ylim([0 100]);
yticks(0:20:100);
title('Random Forest Accuracy');
legend({'Real Data', 'Synthetic Data', 'Combined'}, 'Location', 'southeast');
xticks(logx);
xticklabels(string(file_prefixes));
set(gca, 'FontSize', 12);
set(gcf, 'color', 'w');

% --- Save Results ---
datasetTypes = {'Real', 'Synthetic', 'Combined'};
save('RF_Accuracy_Results.mat', 'allAccuracies', 'realStd', 'syntheticStd', 'combinedStd', 'file_prefixes');
save('RF_Saltwise_Accuracy.mat', 'saltAccuracyMatrix', 'saltNames', 'file_prefixes', 'datasetTypes');

%% ---------------------- Local Function ----------------------

function [trainData, trainLabels, testData, testLabels] = preprocessRF(trainTable, testMetricsOrig, testCategoriesOrig)
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
