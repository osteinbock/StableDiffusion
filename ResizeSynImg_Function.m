function outFilename = resizeSynImgOli(THRES, NUMWHTPIXTARGET, origFilename, inFolder, outFolder)
%RESIZESYNIMG Rescales a grayscale salt–stain image to match a target white-pixel count
%   outFilename = resizeSynImg(THRES, NUMWHTPIXTARGET, origFilename, inFolder, outFolder)
%
%   If NUMWHTPIXTARGET <= 0, we produce a blank canvas (no white pixels).

    % ensure output folder exists
    if ~exist(outFolder,'dir')
        mkdir(outFolder);
    end

    % read & grayscale
    inPath = fullfile(inFolder, origFilename);
    I = imread(inPath);
    if ndims(I)==3, I = rgb2gray(I); end
    I = double(I);

    % original size & background fill
    [H, W] = size(I);
    top    = I(1, :);
    bottom = I(end, :);
    left   = I(:, 1);
    right  = I(:, end);
    bgVal  = median([top(:); bottom(:); left(:); right(:)]);

    % if target is zero or negative → blank canvas and return
    if NUMWHTPIXTARGET <= 0
        canvas      = ones(H, W) * bgVal;
        output      = canvas;
        outFilename = fullfile(outFolder, origFilename);
        imwrite(uint8(output), outFilename);
        fprintf('Target=0 → saved blank canvas: %s\n', outFilename);
        return
    end

    % count current white pixels
    BW   = I > THRES;
    curr = nnz(BW);

    % compute scale
    fudge=9999;
    while fudge<0.8 || fudge>1.2
        fudge=randn(1)*0.2+1;
    end
    
    alpha = sqrt(fudge*NUMWHTPIXTARGET / curr);
    fprintf('Rescaling %s by factor %.4f×\n', origFilename, alpha);

    % rescale
    I_resized = imresize(I, alpha, 'bicubic', 'Antialiasing', true);

    % place or crop
    [h, w] = size(I_resized);
    if h <= H && w <= W
        canvas = ones(H, W) * bgVal;
        row0   = floor((H - h)/2)+1;
        col0   = floor((W - w)/2)+1;
        canvas(row0:row0+h-1, col0:col0+w-1) = I_resized;
        output = canvas;
    else
        row0   = floor((h - H)/2)+1;
        col0   = floor((w - W)/2)+1;
        output = I_resized(row0:row0+H-1, col0:col0+W-1);
    end

    % save
    outFilename = fullfile(outFolder, origFilename);
    imwrite(uint8(output), outFilename);
    fprintf('Saved rescaled image as: %s\n', outFilename);
end
