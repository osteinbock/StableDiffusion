clear all;
close all;

file_prefixes = 128 %[1, 2, 4, 6, 8, 16, 32, 64, 128];
numSizes = length(file_prefixes);
numRuns = 3;

% Store mean accuracy for each training size and mode
allAccuracies = zeros(numSizes, 3);
realStd = zeros(numSizes, 1);
syntheticStd = zeros(numSizes, 1);
combinedStd = zeros(numSizes, 1);

% Load test data
testFile = 'Cropped_ExptFile_ExcludingFirst128PerSalt.txt';
testTable = readtable(testFile, 'Delimiter', '\t');
testSalt = testTable{:, 3};
testMetricsOrig = testTable{:, 9:55};
testCategoriesOrig = string(testSalt);

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

        accReal = runXGBoost(trainTableReal, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns);
        accuraciesReal(end+1) = mean(accReal);

        accSynth = runXGBoost(trainTableSynth, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns);
        accuraciesSynth(end+1) = mean(accSynth);

        accCombined = runXGBoost(trainTableCombined, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns);
        accuraciesCombined(end+1) = mean(accCombined);
    end

    if isempty(accuraciesReal)
        warning('No data for prefix %d, skipping...', prefix);
        continue;
    end

    allAccuracies(i, :) = [mean(accuraciesReal), mean(accuraciesSynth), mean(accuraciesCombined)];
    realStd(i) = std(accuraciesReal);
    syntheticStd(i) = std(accuraciesSynth);
    combinedStd(i) = std(accuraciesCombined);

    fprintf('Prefix %3d | Real: %.2f±%.2f%% | Synth: %.2f±%.2f%% | Comb: %.2f±%.2f%%\n', ...
        prefix, allAccuracies(i,1), realStd(i), allAccuracies(i,2), syntheticStd(i), allAccuracies(i,3), combinedStd(i));
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

% Save results
save('results.mat', 'file_prefixes', 'allAccuracies', 'realStd', 'syntheticStd', 'combinedStd');

%%  --- Collect training and validation accuracy over boosting rounds ---

numTrees = 100;
datasetTypes = {'Real', 'Synth', 'Comb'};
colors = {[0.2 0.7 0.2], [0.9 0.3 0.3], [0.3 0.5 0.9]};
smoothWindow = 5;

trainAccXGB = struct('Real', [], 'Synth', [], 'Comb', []);
valAccXGB = struct('Real', [], 'Synth', [], 'Comb', []);

for d = 1:3
    accTrainMat = [];
    accValMat = [];

    for i = 1:numSizes
        prefix = file_prefixes(i);
        if prefix <= 8
            reps = 1:3;
        else
            reps = 1;
        end

        for rep = reps
            realFile = sprintf('exptData%dImages_%d_Unscored.txt', prefix, rep);
            synthFile = sprintf('ts%d_%d_Data_unscored.txt', prefix, rep);

            if ~isfile(realFile) || ~isfile(synthFile)
                continue
            end

            switch d
                case 1
                    trainTable = readtable(realFile, 'Delimiter', '\t');
                case 2
                    trainTable = readtable(synthFile, 'Delimiter', '\t');
                case 3
                    trainTable = [readtable(realFile, 'Delimiter', '\t'); readtable(synthFile, 'Delimiter', '\t')];
            end

            [trainData, trainLabels, testData, testLabels] = preprocess(trainTable, testMetricsOrig, testCategoriesOrig);

            [labelCats, ~, trainLabelsNum] = unique(trainLabels);
            [~, testLabelsNum] = ismember(testLabels, labelCats);
            trainLabelsNum = trainLabelsNum - 1;
            testLabelsNum = testLabelsNum - 1;

            dtrain = xgb.DMatrix(py.numpy.array(trainData), pyargs('label', py.numpy.array(trainLabelsNum)));
            dtest = xgb.DMatrix(py.numpy.array(testData), pyargs('label', py.numpy.array(testLabelsNum)));

            params = py.dict(pyargs( ...
                'objective', 'multi:softprob', ...
                'num_class', int32(numel(labelCats)), ...
                'max_depth', int32(6), ...
                'eta', 0.3, ...
                'eval_metric', 'merror', ...
                'verbosity', int32(0)));

            num_round = int32(numTrees);
            watchlist = py.list({py.tuple({dtrain, 'train'}), py.tuple({dtest, 'eval'})});
            evals_result = py.dict;
            model = xgb.train(params, dtrain, num_round, pyargs( ...
                'evals', watchlist, ...
                'evals_result', evals_result, ...
                'verbose_eval', false));
            
            trainErrors = double(np.array(evals_result{'train'}{'merror'}));
            valErrors   = double(np.array(evals_result{'eval'}{'merror'}));
            
            trainAcc = (1 - trainErrors) * 100;
            valAcc = (1 - valErrors) * 100;
            
            accTrainMat = [accTrainMat, trainAcc(:)];
            accValMat = [accValMat, valAcc(:)];

        end
    end

    switch d
        case 1
            trainAccXGB.Real = accTrainMat;
            valAccXGB.Real = accValMat;
        case 2
            trainAccXGB.Synth = accTrainMat;
            valAccXGB.Synth = accValMat;
        case 3
            trainAccXGB.Comb = accTrainMat;
            valAccXGB.Comb = accValMat;
    end
end

% --- Plot Training vs Validation Accuracy over boosting rounds ---
figure;
for d = 1:3
    subplot(1,3,d); hold on;
    switch d
        case 1
            trainMat = trainAccXGB.Real;
            valMat = valAccXGB.Real;
            clr = colors{1};
            tit = 'Real Data';
        case 2
            trainMat = trainAccXGB.Synth;
            valMat = valAccXGB.Synth;
            clr = colors{2};
            tit = 'Synthetic Data';
        case 3
            trainMat = trainAccXGB.Comb;
            valMat = valAccXGB.Comb;
            clr = colors{3};
            tit = 'Combined Data';
    end

    validCols = all(~isnan(trainMat),1) & all(trainMat ~= 0,1);
    trainMat = trainMat(:, validCols);
    valMat = valMat(:, validCols);

    meanTrain = mean(trainMat, 2);
    stdTrain = std(trainMat, 0, 2);
    meanVal = mean(valMat, 2);
    stdVal = std(valMat, 0, 2);

    meanTrain = smoothdata(meanTrain, 'movmean', smoothWindow);
    stdTrain = smoothdata(stdTrain, 'movmean', smoothWindow);
    meanVal = smoothdata(meanVal, 'movmean', smoothWindow);
    stdVal = smoothdata(stdVal, 'movmean', smoothWindow);

    trees = 1:numTrees;

    % Ensure all vectors are the correct length
    meanTrain = smoothdata(meanTrain(:), 'movmean', smoothWindow);
    stdTrain = smoothdata(stdTrain(:), 'movmean', smoothWindow);
    meanVal = smoothdata(meanVal(:), 'movmean', smoothWindow);
    stdVal = smoothdata(stdVal(:), 'movmean', smoothWindow);
    
    minLen = min([length(trees), length(meanTrain), length(stdTrain)]);
    
    % Truncate to match
    trees = trees(1:minLen);
    meanTrain = meanTrain(1:minLen);
    stdTrain = stdTrain(1:minLen);
    meanVal = meanVal(1:minLen);
    stdVal = stdVal(1:minLen);
    
    fill([trees fliplr(trees)], ...
         [meanTrain - stdTrain; flipud(meanTrain + stdTrain)]', ...
         clr, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(trees, meanTrain, '-', 'LineWidth', 2, 'Color', clr);
    
    fill([trees fliplr(trees)], ...
         [meanVal - stdVal; flipud(meanVal + stdVal)]', ...
         clr, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(trees, meanVal, '--', 'LineWidth', 2, 'Color', clr);


    xlabel('Number of Boosting Rounds');
    ylabel('Accuracy (%)');
    title(tit);
    ylim([0 100]);
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1);

    if d == 3
        legend({'Train ±1σ', 'Train mean', 'Val ±1σ', 'Val mean'}, ...
            'Location', 'southwest', 'FontSize', 9, 'Box', 'off');
    end
end
set(gcf, 'Color', 'w', 'Units', 'inches', 'Position', [1 1 12 4]);

%% ----------- Local Functions -----------

function accuracies = runXGBoost(trainTable, testMetricsOrig, testCategoriesOrig, xgb, np, numRuns)
    accuracies = zeros(numRuns, 1);
    [trainData, trainLabels, testData, testLabels] = preprocess(trainTable, testMetricsOrig, testCategoriesOrig);

    if any(~isfinite(trainData(:))) || any(~isfinite(testData(:)))
        error('Data contains Inf, NaN, or extreme values!');
    end

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
        model = xgb.train(params, dtrain, num_round, pyargs( ...
            'verbose_eval', false));
        preds = double(model.predict(dtest));
        predictedLabels = categorical(cellstr(labelCats(preds + 1)));
        accuracies(run) = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
    end
end

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
