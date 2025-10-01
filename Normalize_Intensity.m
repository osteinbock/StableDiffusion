% scaleToExpAnchors.m
% -------------------------------------------------------------
% Rescale synthetic images so that their "low" and "high" reference
% intensities match those of the experimental NH4Cl set.
%
%   Experimental folder : C:\...\NH4Cl
%   Synthetic  folder   : C:\...\NH4Clrescaled
%   Output     folder   : C:\...\TESTTEST
% -------------------------------------------------------------

clear; clc;

% -----  USER PATHS -------------------------------------------------------
expFolder = 'D:\StableDiffusionModel\Repeats_on_SmallTrainingSamples\Dreambooth090325_6Images\3\KNO3';
synFolder = 'D:\StableDiffusionModel\Repeats_on_SmallTrainingSamples\Dreambooth090325_6Images\3\KNO3rescaled';
outFolder = 'D:\StableDiffusionModel\Repeats_on_SmallTrainingSamples\Dreambooth090325_6Images\3\KNO3final';

pctBright = 0.05;            % brightest fraction for the "high" anchor
% ------------------------------------------------------------------------

%% 1)  Compute medLowExp and medHighExp from experimental images
[medLowExp, medHighExp] = anchorMedians(expFolder, pctBright);

fprintf('Experimental anchors: medLow = %.2f, medHigh = %.2f\n', ...
         medLowExp, medHighExp);

%% 2)  Prepare output folder
if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

%% 3)  Process each synthetic image
synFiles = [ dir(fullfile(synFolder,'*.png')) ; ...
             dir(fullfile(synFolder,'*.jpg')) ];

if isempty(synFiles)
    error('No PNG or JPG images found in %s', synFolder);
end

for k = 1:numel(synFiles)
    fname = synFiles(k).name;
    I     = imread(fullfile(synFolder, fname));
    if ndims(I)==3, I = rgb2gray(I); end
    I = im2uint8(I);

    % --- anchors for this synthetic image ---
    [medLowSyn, medHighSyn] = anchorForOne(I, pctBright);

    % guard against degenerate case (flat image)
    if medHighSyn <= medLowSyn
        warning('%s: high <= low (skipped)', fname);
        I_scaled = I;    % no change
    else
        % --- linear scaling ---
        scale = (medHighExp - medLowExp) / double(medHighSyn - medLowSyn);
        I_scaled = ( double(I) - double(medLowSyn) ) * scale + medLowExp;
        % clip & cast
        I_scaled = uint8( max(0, min(255, round(I_scaled))) );
    end

    % --- write out ---
    imwrite(I_scaled, fullfile(outFolder, fname));
end

fprintf('Finished. Rescaled images saved to %s\n', outFolder);

%% -------- local helper functions ---------------------------------------
function [medLow, medHigh] = anchorMedians(folder, pctBright)
    % Accumulates median-Low and median-High for every image, then returns
    % the overall medians across the set.
    files = [dir(fullfile(folder,'*.png')) ; dir(fullfile(folder,'*.jpg'))];
    if isempty(files)
        error('No image files in %s', folder);
    end
    n = numel(files);
    lows  = zeros(n,1);
    highs = zeros(n,1);
    for i = 1:n
        I = imread(fullfile(folder, files(i).name));
        if ndims(I)==3, I = rgb2gray(I); end
        I = im2uint8(I);
        [lows(i), highs(i)] = anchorForOne(I, pctBright);
    end
    medLow  = median(lows);
    medHigh = median(highs);
end

function [medLow, medHigh] = anchorForOne(I, pctBright)
    % Returns the border median (medLow) and bright‑pctBright median
    % (medHigh) for a single grayscale image I (uint8).
    % --- border ---
    border = [ I(1,:) , I(end,:) , I(2:end-1,1).' , I(2:end-1,end).' ];
    medLow = median(border);

    % --- brightest pctBright ---
    pix  = double(I(:));
    cut  = prctile(pix, 100*(1-pctBright));
    bright = pix(pix >= cut);
    % trim overshoot to exact fraction
    nTarget = round(pctBright * numel(pix));
    if numel(bright) > nTarget
        bright = bright(1:nTarget);
    end
    medHigh = median(bright);
end
