clear all; 
close all;

filePrefixes = {'ts2', 'ts8', 'ts64'};
% outputFolder = 'PCA_Plots_Shared128';
% if ~exist(outputFolder, 'dir')
%     mkdir(outputFolder);
% end

perSaltVolumes = containers.Map;
trainingSizes = [];

% =============================
% 1. Train PCA on ExptData128_Unscored_061225.txt (renamed variable)
% =============================
train_expt128 = 'ExptData128_Unscored_061225.txt';  
T128 = readtable(train_expt128, 'Delimiter', '\t');
salt128 = string(T128{:, 3});
metrics128 = T128{:, 9:55};

% Remove NaNs
nan_idx_128 = any(isnan(metrics128), 2);
metrics128 = metrics128(~nan_idx_128, :);
salt128 = salt128(~nan_idx_128);
T128 = T128(~nan_idx_128, :);

% Use all of ExptData128 for training PCA (no splitting)
trainMetrics128 = metrics128;

% Z-score
mu = mean(trainMetrics128);
sigma = std(trainMetrics128, 0, 1);
valid_cols = ~(isnan(sigma) | sigma == 0);
trainZ128 = (trainMetrics128(:, valid_cols) - mu(valid_cols)) ./ sigma(valid_cols);

% PCA
[coeff, ~, ~, ~, explained] = pca(trainZ128);

% colors = lines(length(unique(salt128)));  % for plotting
% colors = jet(length(unique(salt128)));
uniqueSalts = unique(salt128);
uniqueSalts = ["NaNO3", "KNO3", "NaCl", "Na2SO3", "Na2SO4", "KCl", "NH4Cl"];
colors = [
    0.8, 0.0, 0.0;     % Dark Red        → for NaNO3
    0.2, 0.4, 1.0;     % Royal Blue      → for KNO3
    0.4, 0.6, 0.0;     % Olive Green     → for NaCl
    1.0, 0.7, 0.0;     % Mustard Yellow  → for Na2SO3
    0.0, 0.0, 0.0;     % Black           → for Na2SO4
    0.0, 0.9, 0.9;     % Cyan            → for KCl
    1.0, 0.4, 0.7      % Pink            → for NH4Cl
];




% Map raw salt names to display-friendly strings with subscripts
saltLabelMap = containers.Map( ...
    {'KCl', 'KNO3', 'NaNO3', 'Na2SO3', 'Na2SO4', 'NaCl', 'NH4Cl'}, ...
    {'KCl', 'KNO_3', 'NaNO_3', 'Na_2SO_3', 'Na_2SO_4', 'NaCl', 'NH_4Cl'} ...
);

% =============================
% Calculate PCA scores of training data once
% =============================
scoreTrain128 = trainZ128 * coeff;

% =============================
% 2. Project all datasets fully onto the trained PCA space
% =============================
allScores = [];  % For axis limits

allScoresPerPrefix = cell(length(filePrefixes),1);
allSaltPerPrefix = cell(length(filePrefixes),1);
trainingSizes = zeros(length(filePrefixes),1);

for k = 1:length(filePrefixes)
    prefix = filePrefixes{k};
    file = sprintf('%s_Data_unscored.txt', prefix);
    trainSize = str2double(regexprep(prefix, '[^\d]', ''));
    trainingSizes(k) = trainSize;

    fprintf('Processing %s...\n', file);

    try
        % Load and preprocess
        T = readtable(file, 'Delimiter', '\t');
        salt = string(T{:, 3});
        metrics = T{:, 9:55};
        nan_idx = any(isnan(metrics), 2);
        metrics = metrics(~nan_idx, :);
        salt = salt(~nan_idx);
        T = T(~nan_idx, :);

        % Z-score using PCA training mean/std
        Z = (metrics(:, valid_cols) - mu(valid_cols)) ./ sigma(valid_cols);
        scoreAll = Z * coeff;
        % Limit the number of test samples per salt to 128 (same as training)
        maxSamplesPerSalt = 128;
        limited_idx = false(size(salt));
        for iSalt = 1:length(uniqueSalts)
            sName = uniqueSalts(iSalt);
            idx = find(salt == sName);
            if numel(idx) > maxSamplesPerSalt
                idx = idx(1:maxSamplesPerSalt);  % alternatively: randperm for random
            end
            limited_idx(idx) = true;
        end
        % Apply filtering
        salt = salt(limited_idx);
        scoreAll = scoreAll(limited_idx, :);


        % Collect all scores for axis limits
        allScores = [allScores; scoreAll(:,1:3)];

        allScoresPerPrefix{k} = scoreAll;
        allSaltPerPrefix{k} = salt;

    catch ME
        warning('Error processing %s: %s', prefix, ME.message);
        allScoresPerPrefix{k} = [];
        allSaltPerPrefix{k} = [];
    end
end

% =============================
% 3. Determine common axis limits for all subplots
% =============================
axisMargin = 0.1;  % 10% margin

minVals = min(allScores, [], 1);
maxVals = max(allScores, [], 1);
rangeVals = maxVals - minVals;
xlim_vals = [-12.5,12.5]; %[minVals(1) - axisMargin*rangeVals(1), maxVals(1) + axisMargin*rangeVals(1)];
ylim_vals = [minVals(2) - axisMargin*rangeVals(2), maxVals(2) + axisMargin*rangeVals(2)];
zlim_vals = [minVals(3) - axisMargin*rangeVals(3), maxVals(3) + axisMargin*rangeVals(3)];

% =============================
% 4. Plot all datasets with common axes, add training size label, comment out titles
% =============================
figure('Color', 'w', 'Position', [100, 100, 1400, 900]);
for k = 1:length(filePrefixes)
    prefix = filePrefixes{k};
    trainSize = trainingSizes(k);

    scoreAll = allScoresPerPrefix{k};
    salt = allSaltPerPrefix{k};

    if isempty(scoreAll)
        continue;
    end

    subplot(1, 4, k+1);
    hold on;
    for i = 1:length(uniqueSalts)
        idx = salt == uniqueSalts(i);
        saltName = char(uniqueSalts(i));
        labelStr = saltLabelMap(saltName);
        scatter(scoreAll(idx,1), scoreAll(idx,2), ...
                12, colors(i,:), 'filled', 'DisplayName', labelStr);
    end
    hold off;
    grid on;
    xlabel('PC1', 'FontWeight', 'bold');
    ylabel('PC2', 'FontWeight', 'bold');
    % title(sprintf('All Data Projected in ExptData128 PCA (%s)', prefix), 'Interpreter', 'none'); % Commented out

    xlim(xlim_vals);
    ylim(ylim_vals);
    axis square

    ax = gca;          % Increase tick label font size
    ax.FontSize = 12;

    text(xlim_vals(1) + 0.05*diff(xlim_vals), ylim_vals(2) - 0.1*diff(ylim_vals), ...
         prefix, 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'w');

    if k == 1
        legend('show', 'Location', 'bestoutside', 'FontSize', 10);
    end
end

% =============================
% 6. Plot PCA training data as final subplot 
% =============================
subplot(1, 4, 1);
hold on;
for i = 1:length(uniqueSalts)
    idx = salt128 == uniqueSalts(i);
    scatter(scoreTrain128(idx,1), scoreTrain128(idx,2), ...
        12, colors(i,:), 'filled', 'DisplayName', uniqueSalts(i));
end
hold off;
grid on;
xlabel(sprintf('PC1 (%.2f%%)', explained(1)), 'FontWeight', 'bold');
ylabel(sprintf('PC2 (%.2f%%)', explained(2)), 'FontWeight', 'bold');
title('Training Data', 'FontWeight', 'bold');
xlim(xlim_vals);
ylim(ylim_vals);
axis('square')

