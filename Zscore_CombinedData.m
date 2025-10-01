% Combine and Z-score experimental and synthetic data
% 09/07/25

clear all
close all

% -------------------------------------------------------------------------
% Step 1: Read experimental data
expFile = 'exptData128Images_Unscored.txt';
expData = readtable(expFile, 'Delimiter', '\t');

% Mark as Experiment, then move to 4th column
expData.Source = repmat({'Experiment'}, height(expData), 1);
expData = movevars(expData, 'Source', 'After', expData.Properties.VariableNames{3});  % -> col 4

% -------------------------------------------------------------------------
% Step 2: Read synthetic data
synFile = 'ts128_Data_unscored.txt';
synData = readtable(synFile, 'Delimiter', '\t');

% Mark as Synthetic, then move to 4th column
synData.Source = repmat({'Synthetic'}, height(synData), 1);
synData = movevars(synData, 'Source', 'After', synData.Properties.VariableNames{3});  % -> col 4

% -------------------------------------------------------------------------
% Step 3: Select only the first 128 synthetic images for each salt
% Assumption: Column 3 contains SaltName
[uniqueSalts, ~, saltIdx] = unique(synData{:,3}, 'stable');

selectedSynData = [];
for i = 1:numel(uniqueSalts)
    rows = find(saltIdx == i);
    selectedRows = rows(1:min(128, numel(rows)));
    selectedSynData = [selectedSynData; synData(selectedRows,:)];
end

% -------------------------------------------------------------------------
% Step 4: Combine experiment and synthetic datasets
combinedData = [expData; selectedSynData];

% -------------------------------------------------------------------------
% Step 5: Z-score the metric columns (globally, like original script)
% Metrics are columns 11 to 57
metricCols = 10:56;
numericData = combinedData{:, metricCols};

zScoredData = numericData * 0;
for i = 1:size(numericData,2)
    zScoredData(:,i) = (numericData(:,i) - mean(numericData(:,i), 'omitnan')) ./ ...
                       std(numericData(:,i), 'omitnan');
end

% Create z-scored table: keep cols 1–10 as-is, replace 11–57 with z-scored
zScoredTable = [combinedData(:, 1:9), ...
                array2table(zScoredData, 'VariableNames', combinedData.Properties.VariableNames(metricCols))];

% -------------------------------------------------------------------------
% Step 6: Save output
writetable(zScoredTable, 'combinedData_ExptSyn_128.txt', 'Delimiter', '\t');

% -------------------------------------------------------------------------
% Step 7: Visualize (optional)
figure;
imagesc(zScoredData, [-10 10]);
colormap(hot);
colorbar;
title('Z-scored Metrics (Experiment + Synthetic)');
xlabel('Metrics (10–56)');
ylabel('Images');
