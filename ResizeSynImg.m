%==========================================================================
% processAllSalts_052125.m
%
% 1) Plot all seven empirical vs. bootstrap histograms (2×4 layout),
%    truncating each salt’s data to [0.5*μ, 2.0*μ] before sampling.
% 2) Ask once “Continue processing images? Y/N”
% 3) If Y, for each salt:
%     • bootstrap N targets
%     • read PNGs from "<salt>best/"
%     • create "<salt>rescaled/" if needed
%     • call resizeSynImg(75, target_j, fname_j, inFldr, outFldr)
%==========================================================================

clear; clc;

THRES = 100;     % grayscale threshold for numWhitePixel / important also for size

rng(1234);      % random number seed for reproducibility

%% 1) Read data and detect columns
fname = 'exptData6Images_Unscored_3rdSet.txt';
opts  = detectImportOptions(fname, 'FileType','text','Delimiter','\t');
T     = readtable(fname, opts);

vn      = T.Properties.VariableNames;
saltCol = vn{ find(contains(lower(vn),'salt'),     1) };
pixCol  = vn{ find(contains(lower(vn),'numwhite'), 1) };

T.(saltCol) = categorical(T.(saltCol));
saltNames   = categories(T.(saltCol));
nSalts      = numel(saltNames);

%% 2) Truncate, bootstrap & plot all in one figure
bootstrapSamples = cell(nSalts,1);

figure('Name','Empirical vs. Bootstrap Distributions','Color','w');
for k = 6:6 %nSalts
    salt = saltNames{k};
    data_k = T.(pixCol)(T.(saltCol)==salt);
    mu     = mean(data_k);

    % truncate to [0.5*μ, 2.0*μ]
    lo    = 0.99*mu;
    hi    = 1.01*mu;
    trunc = data_k(data_k >= lo & data_k <= hi);
    if isempty(trunc)
        trunc = data_k;
        warning('Salt %s: truncation empty, using full data.', salt);
    end

    % count images in "<salt>best"
    inFldr = salt + "best";
    imgs   = dir(fullfile(inFldr,'*.png'));
    N      = numel(imgs);
    assert(N>0, 'No PNGs found in %s.', inFldr);

    % bootstrap N samples
    samples = datasample(trunc, N);
    bootstrapSamples{k} = samples;

    % plot overlay
    subplot(2,4,k);
    hEmp = histogram(data_k, 'Normalization','count', ...
                     'EdgeColor','none','FaceAlpha',0.6);
    hold on;
    histogram(samples, hEmp.BinEdges, 'Normalization','count', ...
              'EdgeColor','none','FaceAlpha',0.4);
    hold off;
    title(salt, 'Interpreter','none');
    xlabel(pixCol); ylabel('Count');
    legend('Empirical','Bootstrap','Location','northeast');
    grid on;
end
subplot(2,4,8); axis off;
sgtitle('Empirical vs. Bootstrap (Truncated) by Salt','FontSize',14);

%% 3) Prompt to continue
resp = input('Continue processing images for all salts? Y/N [Y]: ','s');
if ~isempty(resp) && upper(resp)=='N'
    fprintf('Aborted by user.\n');
    return;
end
close;

%% 4) Process each salt using its bootstrap targets
for k = 6:6 %nSalts
    salt    = saltNames{k};
    inFldr  = salt + "best";
    outFldr = salt + "rescaled";
    samples = bootstrapSamples{k};

    imgs = dir(fullfile(inFldr,'*.png'));
    N    = numel(imgs);

    fprintf('\n--- Processing %s (%d images) ---\n', salt, N);
    if ~exist(outFldr,'dir')
        mkdir(outFldr);
    end

    for j = 1:N
        fname_j = imgs(j).name;
        target  = samples(j);
        fprintf(' [%d/%d] %s → target %d\n', j, N, fname_j, target);
        resizeSynImgOli(THRES, target, fname_j, inFldr, outFldr);
    end
end

fprintf('\nAll salts processed.\n');
