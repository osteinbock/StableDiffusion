% select_exactly_676_split.m
% Reads all .png in inputDir, rejects images by:
%   1) too-dark (global mean gray < grayThresh)
%   2) too-colorful (mean saturation > satThresh)
%   3) of the remaining, reject exactly R = N_remain – targetN images:
%        • nBright = round(brightSplit * R) brightest-border by mean
%        • nNoisy  = R – nBright                   noisiest-border by std
% Copies the targetN “keepers” into outputDir and verifies count.

clear; clc;

%% Parameters
targetN    = 676;      % desired number of keepers
w          = 20;       % border width in pixels
satThresh  = 0.3;     % mean-saturation threshold [0–1] (0.05 for Na2SO3, 0.2 otherwise)
thumbScale = 0.1;      % thumbnail scale for color check
grayThresh = 10;       % global mean gray cutoff [0–255] (excludes black squares)
brightSplit = 0.4;    % fraction of rejects to attribute to brightness (0.7 for Na2SO3, 0.4 otherwise)

inputDir   = 'NH4ClFrom6Image';
outputDir  = 'NH4Clbest';

%% 1) List files & show count
files  = dir(fullfile(inputDir, '*.png'));
nFiles = numel(files);
fprintf('Found %d PNG files in "%s"\n\n', nFiles, inputDir);

%% 2) Prepare output folder
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% 3) Preallocate stats
borderMean = zeros(nFiles,1);
borderStd  = zeros(nFiles,1);
isDark     = false(nFiles,1);
isColor    = false(nFiles,1);

%% 4) Compute per-image stats
for i = 1:nFiles
    % Read image
    Iorig = imread(fullfile(files(i).folder, files(i).name));
    
    % -- Dark test on global gray --
    if ndims(Iorig) == 3
        gray = rgb2gray(Iorig);
    else
        gray = Iorig;
    end
    isDark(i) = mean(gray(:)) < grayThresh;
    
    % -- Border mean & std (normalized grayscale) --
    G = im2double(gray);
    top    = G(1:w,       :);
    bottom = G(end-w+1:end, :);
    left   = G(w+1:end-w,   1:w);
    right  = G(w+1:end-w,   end-w+1:end);
    vals   = [top(:); bottom(:); left(:); right(:)];
    borderMean(i) = mean(vals);
    borderStd(i)  = std(vals);
    
    % -- Color detection via thumbnail saturation --
    if ndims(Iorig) == 3
        thumb      = imresize(Iorig, thumbScale, 'bilinear');
        hsvThumb   = rgb2hsv(thumb);
        isColor(i) = mean2(hsvThumb(:,:,2)) > satThresh;
    end
end

%% 5) Reject dark & color
idx_dark  = find(isDark);
rem1      = setdiff(1:nFiles, idx_dark);
idx_color = intersect(rem1, find(isColor));
rem2      = setdiff(rem1, idx_color);

%% 6) Compute how many to reject to hit targetN
nRemains = numel(rem2);
R = nRemains - targetN;
if R < 0
    error('Not enough images (%d) after dark/color to reach %d.', nRemains, targetN);
end

%% 7) Reject nBright brightest-border by mean
nBright    = round(brightSplit * R);
[~, ordB]  = sort(borderMean(rem2), 'descend');
idx_bright = rem2(ordB(1:nBright));

%% 8) Reject nNoisy noisiest-border by std from the rest
rem3       = setdiff(rem2, idx_bright);
nNoisy     = R - nBright;
[~, ordN]  = sort(borderStd(rem3), 'descend');
idx_noisy  = rem3(ordN(1:nNoisy));

%% 9) Copy keepers
rejectIdx = unique([idx_dark(:); idx_color(:); idx_bright(:); idx_noisy(:)]);
keeperIdx = setdiff(1:nFiles, rejectIdx);

for k = keeperIdx
    src = fullfile(files(k).folder, files(k).name);
    dst = fullfile(outputDir, files(k).name);
    copyfile(src, dst);
end

%% 10) Report & verify
fprintf('Rejected:\n');
fprintf('  Too-dark      : %4d\n', numel(idx_dark));
fprintf('  Too-colorful  : %4d\n', numel(idx_color));
fprintf('  Bright-border : %4d\n', numel(idx_bright));
fprintf('  Noisy-border  : %4d\n', numel(idx_noisy));
fprintf('---------------------------\n');
fprintf('Total copied to "%s": %4d (target %d)\n\n', ...
        outputDir, numel(keeperIdx), targetN);

assert(numel(keeperIdx) == targetN, ...
       'Error: expected %d keepers, but got %d.', targetN, numel(keeperIdx));
