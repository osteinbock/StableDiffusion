clear all;
close all;

file_prefixes = [1, 2, 4, 6, 8, 16, 32, 64, 128];
numSizes = length(file_prefixes);
numRuns = 3;

% Store mean accuracy for each training size and mode
allAccuracies = zeros(numSizes, 3);
syntheticStd = zeros(numSizes, 1);
combinedStd = zeros(numSizes, 1);
realStd = zeros(numSizes, 1);

maxEpochs = 100;

trainAccReal = nan(maxEpochs, 3, numSizes);
valAccReal = nan(maxEpochs, 3, numSizes);
trainAccSynth = nan(maxEpochs, 3, numSizes);
valAccSynth = nan(maxEpochs, 3, numSizes);
trainAccComb = nan(maxEpochs, 3, numSizes);
valAccComb = nan(maxEpochs, 3, numSizes);

% Load test data (fixed)
testFile = 'Cropped_ExptFile_ExcludingFirst128PerSalt.txt';
testTable = readtable(testFile, 'Delimiter', '\t');
testTable{isinf(testTable{:,30}), 30} = mean(testTable{~isinf(testTable{:,30}), 30});
testSalt = testTable{:, 3};
testMetricsOrig = testTable{:, 9:55};
testCategoriesOrig = string(testSalt);

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

    repCounter = 1;
    for rep = reps
        realFile = sprintf('exptData%dImages_%d_Unscored.txt', prefix, rep);
        trainTableReal = readtable(realFile, 'Delimiter', '\t');

        synthFile = sprintf('ts%d_%d_Data_unscored.txt', prefix, rep);
        trainTableSynth = readtable(synthFile, 'Delimiter', '\t');

        % === REAL ===
        runAccuracies = zeros(numRuns,1);
        [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
            preprocess(trainTableReal, testMetricsOrig, testCategoriesOrig);

        for run = 1:numRuns
            layers = getMLPLayers(size(trainData, 2), numel(categories(trainLabels)));
            options = getTrainingOptions(valData, valLabels);

            [net, trainInfo] = trainNetwork(trainData, trainLabels, layers, options);

            predicted = classify(net, testData);
            runAccuracies(run) = mean(predicted == testLabels) * 100;

            trainAccReal(1:length(trainInfo.TrainingAccuracy), repCounter, i) = trainInfo.TrainingAccuracy;
            valAccReal(1:length(trainInfo.ValidationAccuracy), repCounter, i) = trainInfo.ValidationAccuracy;
        end
        accuraciesReal(end+1) = mean(runAccuracies);

        % === SYNTHETIC ===
        runAccuracies = zeros(numRuns,1);
        [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
            preprocess(trainTableSynth, testMetricsOrig, testCategoriesOrig);

        for run = 1:numRuns
            layers = getMLPLayers(size(trainData, 2), numel(categories(trainLabels)));
            options = getTrainingOptions(valData, valLabels);

            [net, trainInfo] = trainNetwork(trainData, trainLabels, layers, options);

            predicted = classify(net, testData);
            runAccuracies(run) = mean(predicted == testLabels) * 100;

            trainAccSynth(1:length(trainInfo.TrainingAccuracy), repCounter, i) = trainInfo.TrainingAccuracy;
            valAccSynth(1:length(trainInfo.ValidationAccuracy), repCounter, i) = trainInfo.ValidationAccuracy;
        end
        accuraciesSynth(end+1) = mean(runAccuracies);

        % === COMBINED ===
        trainTableCombined = [trainTableReal; trainTableSynth];
        runAccuracies = zeros(numRuns,1);
        [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
            preprocess(trainTableCombined, testMetricsOrig, testCategoriesOrig);

        for run = 1:numRuns
            layers = getMLPLayers(size(trainData, 2), numel(categories(trainLabels)));
            options = getTrainingOptions(valData, valLabels);

            [net, trainInfo] = trainNetwork(trainData, trainLabels, layers, options);

            predicted = classify(net, testData);
            runAccuracies(run) = mean(predicted == testLabels) * 100;

            trainAccComb(1:length(trainInfo.TrainingAccuracy), repCounter, i) = trainInfo.TrainingAccuracy;
            valAccComb(1:length(trainInfo.ValidationAccuracy), repCounter, i) = trainInfo.ValidationAccuracy;
        end
        accuraciesCombined(end+1) = mean(runAccuracies);

        repCounter = repCounter + 1;
    end

    allAccuracies(i, 1) = mean(accuraciesReal);
    allAccuracies(i, 2) = mean(accuraciesSynth);
    allAccuracies(i, 3) = mean(accuraciesCombined);
    realStd(i) = std(accuraciesReal);
    syntheticStd(i) = std(accuraciesSynth);
    combinedStd(i) = std(accuraciesCombined);

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

% --- TRAINING vs VALIDATION ACCURACY PLOTS ---
figure;
datasetTypes = {'Real', 'Synthetic', 'Combined'};
colors = {[0.2 0.7 0.2], [0.9 0.3 0.3], [0.3 0.5 0.9]};
smoothWindow = 5;

for d = 1:3
    subplot(1, 3, d); hold on;

    switch d
        case 1
            trainAcc = trainAccReal; valAcc = valAccReal;
        case 2
            trainAcc = trainAccSynth; valAcc = valAccSynth;
        case 3
            trainAcc = trainAccComb; valAcc = valAccComb;
    end

    [E, R, P] = size(trainAcc);
    trainFlat = reshape(trainAcc, E, R*P);
    valFlat   = reshape(valAcc, E, R*P);

    validCols = any(~isnan(trainFlat),1) & any(trainFlat~=0,1);
    trainFlat = trainFlat(:,validCols);
    valFlat   = valFlat(:,validCols);

    meanTrain = smoothdata(nanmean(trainFlat,2), 'movmean', smoothWindow);
    stdTrain  = nanstd(trainFlat, 0, 2);
    meanVal   = smoothdata(nanmean(valFlat,2), 'movmean', smoothWindow);
    stdVal    = nanstd(valFlat, 0, 2);

    epochs = (1:E)';
    fill([epochs; flipud(epochs)], [meanTrain-stdTrain; flipud(meanTrain+stdTrain)], ...
        colors{d}, 'FaceAlpha',0.15, 'EdgeColor','none');
    plot(epochs, meanTrain, '-', 'LineWidth', 2, 'Color', colors{d});

    fill([epochs; flipud(epochs)], [meanVal-stdVal; flipud(meanVal+stdVal)], ...
        colors{d}, 'FaceAlpha',0.15, 'EdgeColor','none');
    plot(epochs, meanVal, '--', 'LineWidth', 2, 'Color', colors{d});

    xlabel('Epoch');
    ylabel('Accuracy (%)');
    title([datasetTypes{d} ' Data']);
    ylim([0 100]);
    grid on;
    set(gca,'FontSize',12,'LineWidth',1);
    if d == 3
        legend({'Train ±1σ','Train mean','Val ±1σ','Val mean'}, ...
            'Location','southwest','FontSize',9,'Box','off');
    end
end

set(gcf, 'Color', 'w', 'Units', 'inches', 'Position', [1 1 12 4]);  % Wide figure layout


%% Local functions
function [trainData, trainLabels, valData, valLabels, testData, testLabels] = preprocess(trainTable, testMetricsOrig, testCategoriesOrig)
    % Extract training salt and metrics
    trainSalt = trainTable{:, 3};
    trainMetrics = trainTable{:, 9:55};
    trainCategories = string(trainSalt);

    % Match categories between train and test
    [uniqueCategories, ~, trainLabelsRaw] = unique(trainCategories);
    [~, locb] = ismember(testCategoriesOrig, uniqueCategories);
    validIdx = locb > 0;
    testLabelsRaw = locb(validIdx);
    testMetrics = testMetricsOrig(validIdx, :);

    % Labels
    trainLabels = categorical(trainLabelsRaw);
    testLabels = categorical(testLabelsRaw);

    % Fill missing values
    trainFilled = fillmissing(trainMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));
    testFilled  = fillmissing(testMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));

    % Normalize using training set statistics
    trainMean = mean(trainFilled, 1);
    trainStd = std(trainFilled, 0, 1);
    trainData = (trainFilled - trainMean) ./ trainStd;
    testData = (testFilled - trainMean) ./ trainStd;

    % Split train into train/validation (80/20 stratified)
    cv = cvpartition(trainLabels, 'HoldOut', 0.2);
    idxTrain = training(cv);
    idxVal = test(cv);

    valData = trainData(idxVal, :);
    valLabels = trainLabels(idxVal);
    trainData = trainData(idxTrain, :);
    trainLabels = trainLabels(idxTrain);
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

function options = getTrainingOptions(valData, valLabels)
    options = trainingOptions('adam', ...
        'MiniBatchSize', 128, ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 0.01, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {valData, valLabels}, ...
        'ValidationPatience', 5, ...
        'Verbose', false, ...
        'Plots', 'none');
end

