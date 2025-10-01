clear all;
close all;

file_prefixes = 2; %[1, 2, 4, 6, 8, 16, 32, 64, 128];
numSizes = length(file_prefixes);
numRuns = 3;
numTrees = 100;

% Store mean accuracy for each training size and mode (Real, Synthetic, Combined)
allAccuracies = zeros(numSizes, 3);
realStd = zeros(numSizes, 1);
syntheticStd = zeros(numSizes, 1);
combinedStd = zeros(numSizes, 1);

% Store OOB and Train Accuracy Over Trees
trainAccRF = struct('Real', [], 'Synth', [], 'Comb', []);
valAccRF   = struct('Real', [], 'Synth', [], 'Comb', []);

% Load fixed test data
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

    for rep = reps
        % --- REAL DATA ---
        realFile = sprintf('exptData%dImages_%d_Unscored.txt', prefix, rep);
        trainTableReal = readtable(realFile, 'Delimiter', '\t');

        for run = 1:numRuns
            if prefix > 2
                % Split training into train/validation sets (70:30) to avoid data leakage
                [trainData, trainLabels, valData, valLabels] = preprocessRF_split(trainTableReal);
            else
                % For small datasets, skip splitting; train on all data, val same as train
                [trainData, trainLabels] = preprocessRF_noSplit(trainTableReal);
                valData = trainData;
                valLabels = trainLabels;
            end

            rfModel = TreeBagger(numTrees, trainData, trainLabels, ...
                'Method', 'classification', 'OOBPrediction', 'On');

            % Evaluate on held-out test data (final test accuracy)
            [testData, testLabels] = processTestData(trainTableReal, testMetricsOrig, testCategoriesOrig);

            predicted = predict(rfModel, testData);
            predicted = categorical(predicted);
            acc = mean(predicted == testLabels) * 100;
            accuraciesReal(end+1) = acc;

            % OOB validation accuracy from model
            oobErr = oobError(rfModel);
            valAcc = (1 - oobErr) * 100; % [numTrees x 1]

            % Train accuracy per tree on training data
            trainAcc = zeros(numTrees, 1);
            for t = 1:numTrees
                preds = predict(rfModel.Trees{t}, trainData);
                trainAcc(t) = mean(categorical(preds) == trainLabels) * 100;
            end

            trainAccRF.Real = [trainAccRF.Real, trainAcc(:)];
            valAccRF.Real   = [valAccRF.Real, valAcc(:)];
        end

        % --- SYNTHETIC DATA ---
        synthFile = sprintf('ts%d_%d_Data_unscored.txt', prefix, rep);
        trainTableSynth = readtable(synthFile, 'Delimiter', '\t');

        for run = 1:numRuns
            if prefix > 2
                [trainData, trainLabels, valData, valLabels] = preprocessRF_split(trainTableSynth);
            else
                [trainData, trainLabels] = preprocessRF_noSplit(trainTableSynth);
                valData = trainData;
                valLabels = trainLabels;
            end

            rfModel = TreeBagger(numTrees, trainData, trainLabels, ...
                'Method', 'classification', 'OOBPrediction', 'On');

            [testData, testLabels] = processTestData(trainTableSynth, testMetricsOrig, testCategoriesOrig);

            predicted = predict(rfModel, testData);
            predicted = categorical(predicted);
            acc = mean(predicted == testLabels) * 100;
            accuraciesSynth(end+1) = acc;

            oobErr = oobError(rfModel);
            valAcc = (1 - oobErr) * 100;

            trainAcc = zeros(numTrees, 1);
            for t = 1:numTrees
                preds = predict(rfModel.Trees{t}, trainData);
                trainAcc(t) = mean(categorical(preds) == trainLabels) * 100;
            end

            trainAccRF.Synth = [trainAccRF.Synth, trainAcc(:)];
            valAccRF.Synth   = [valAccRF.Synth, valAcc(:)];
        end

        % --- COMBINED DATA ---
        trainTableCombined = [trainTableReal; trainTableSynth];

        for run = 1:numRuns
            if prefix > 2
                [trainData, trainLabels, valData, valLabels] = preprocessRF_split(trainTableCombined);
            else
                [trainData, trainLabels] = preprocessRF_noSplit(trainTableCombined);
                valData = trainData;
                valLabels = trainLabels;
            end

            rfModel = TreeBagger(numTrees, trainData, trainLabels, ...
                'Method', 'classification', 'OOBPrediction', 'On');

            [testData, testLabels] = processTestData(trainTableCombined, testMetricsOrig, testCategoriesOrig);

            predicted = predict(rfModel, testData);
            predicted = categorical(predicted);
            acc = mean(predicted == testLabels) * 100;
            accuraciesCombined(end+1) = acc;

            oobErr = oobError(rfModel);
            valAcc = (1 - oobErr) * 100;

            trainAcc = zeros(numTrees, 1);
            for t = 1:numTrees
                preds = predict(rfModel.Trees{t}, trainData);
                trainAcc(t) = mean(categorical(preds) == trainLabels) * 100;
            end

            trainAccRF.Comb = [trainAccRF.Comb, trainAcc(:)];
            valAccRF.Comb   = [valAccRF.Comb, valAcc(:)];
        end
    end

    allAccuracies(i,1) = mean(accuraciesReal);
    allAccuracies(i,2) = mean(accuraciesSynth);
    allAccuracies(i,3) = mean(accuraciesCombined);
    realStd(i) = std(accuraciesReal);
    syntheticStd(i) = std(accuraciesSynth);
    combinedStd(i) = std(accuraciesCombined);

    fprintf('Prefix %3d | Real: %.2f%% | Synth: %.2f%% | Comb: %.2f%%\n', ...
        prefix, allAccuracies(i,1), allAccuracies(i,2), allAccuracies(i,3));
end

% --- Test Accuracy Plot ---
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

% --- Train vs Validation Accuracy Over Trees ---
figure;
datasetTypes = {'Real', 'Synth', 'Comb'};
colors = {[0.2 0.7 0.2], [0.9 0.3 0.3], [0.3 0.5 0.9]};
smoothWindow = 5;

for d = 1:3
    subplot(1, 3, d); hold on;
    switch d
        case 1
            trainMat = trainAccRF.Real;
            valMat   = valAccRF.Real;
        case 2
            trainMat = trainAccRF.Synth;
            valMat   = valAccRF.Synth;
        case 3
            trainMat = trainAccRF.Comb;
            valMat   = valAccRF.Comb;
    end

    validCols = any(~isnan(trainMat),1) & any(trainMat ~= 0,1);
    trainMat = trainMat(:,validCols);
    valMat   = valMat(:,validCols);

    meanTrain = mean(trainMat, 2);
    stdTrain  = std(trainMat, 0, 2);
    meanVal   = mean(valMat, 2);
    stdVal    = std(valMat, 0, 2);
    
    meanTrain = smoothdata(meanTrain,'movmean',smoothWindow);
    meanVal   = smoothdata(meanVal,'movmean',smoothWindow);
    stdTrain  = smoothdata(stdTrain,'movmean',smoothWindow);
    stdVal    = smoothdata(stdVal,'movmean',smoothWindow);
    
    trees = 1:numTrees;
    minLen = min([length(trees), length(meanTrain), length(stdTrain), length(meanVal), length(stdVal)]);
    trees = trees(1:minLen);
    meanTrain = meanTrain(1:minLen);
    stdTrain  = stdTrain(1:minLen);
    meanVal   = meanVal(1:minLen);
    stdVal    = stdVal(1:minLen);

    fill([trees(:)' fliplr(trees(:)')], ...
     [(meanTrain(:) - stdTrain(:))' fliplr((meanTrain(:) + stdTrain(:))')], ...
     colors{d}, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(trees, meanTrain, '-', 'LineWidth', 2, 'Color', colors{d});
    
    fill([trees(:)' fliplr(trees(:)')], ...
         [(meanVal(:) - stdVal(:))' fliplr((meanVal(:) + stdVal(:))')], ...
         colors{d}, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(trees, meanVal, '--', 'LineWidth', 2, 'Color', colors{d});

    xlabel('Number of Trees');
    ylabel('Accuracy (%)');
    title([datasetTypes{d} ' Data']);
    ylim([0 100]);
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1);

    if d == 3
        legend({'Train ±1σ','Train mean','Val ±1σ','Val mean'}, ...
            'Location','southwest','FontSize',9,'Box','off');
    end
end

set(gcf,'Color','w','Units','inches','Position',[1 1 12 4]);

%% ---------------------- Local Functions ----------------------

function [X_train, y_train, X_val, y_val] = preprocessRF_split(trainTable)
    % Normalize and split 70:30 train-validation from training table
    
    trainSalt = trainTable{:, 3};
    trainMetrics = trainTable{:, 9:55};
    trainCategories = string(trainSalt);

    [~, ~, labelsRaw] = unique(trainCategories);
    labels = categorical(labelsRaw);

    dataFilled = fillmissing(trainMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));
    meanVals = mean(dataFilled, 1);
    stdVals = std(dataFilled, 0, 1);
    normData = (dataFilled - meanVals) ./ stdVals;

    % 70:30 stratified split by labels
    cv = cvpartition(labels, 'HoldOut', 0.3);
    X_train = normData(training(cv), :);
    y_train = labels(training(cv));
    X_val   = normData(test(cv), :);
    y_val   = labels(test(cv));
end

function [X_train, y_train] = preprocessRF_noSplit(trainTable)
    % For small datasets, no split: use all data as training
    trainSalt = trainTable{:, 3};
    trainMetrics = trainTable{:, 9:55};
    trainCategories = string(trainSalt);

    [~, ~, labelsRaw] = unique(trainCategories);
    labels = categorical(labelsRaw);

    dataFilled = fillmissing(trainMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));
    meanVals = mean(dataFilled, 1);
    stdVals = std(dataFilled, 0, 1);
    normData = (dataFilled - meanVals) ./ stdVals;

    X_train = normData;
    y_train = labels;
end

function [testData, testLabels] = processTestData(trainTable, testMetricsOrig, testCategoriesOrig)
    % Normalize test data based on training data stats
    trainSalt = trainTable{:, 3};
    trainCategories = string(trainSalt);
    [uniqueCategories, ~, ~] = unique(trainCategories);
    [~, locb] = ismember(testCategoriesOrig, uniqueCategories);
    validIdx = locb > 0;
    testLabels = categorical(locb(validIdx));
    testMetrics = testMetricsOrig(validIdx, :);

    trainMetrics = trainTable{:, 9:55};
    trainFilled = fillmissing(trainMetrics, 'constant', mean(trainMetrics, 1, 'omitnan'));

    trainMean = mean(trainFilled, 1);
    trainStd = std(trainFilled, 0, 1);

    testData = (testMetrics - trainMean) ./ trainStd;
end
