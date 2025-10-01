% Dried Salt Drop Analysis - RODI_MainProcessing_062424.m
% Monday 06/25/24 
% Reads Images and Creates Metric Vector Data

clear all
close all
clf

mainDir = 'D:\StableDiffusionModel\Repeats_on_SmallTrainingSamples\Dreambooth090325_6Images\3';
folderList = {'KClfinal', 'KNO3final', 'Na2SO3final', 'Na2SO4final', 'NaClfinal', 'NaNO3final', 'NH4Clfinal'};

for k = 1:length(folderList)
    currentFolder = fullfile(mainDir, folderList{k});
    cd(currentFolder);  % This makes dir('*.png') work for the current folder
    
    % Output Filename
    filename = 'results.txt';
        
        
    % Main Parameters
    THRES=70;               % intensity threshold used to turn grayscale into binary image 
    HIGHTHRES = 200;        % defines bright areas
    THRESTMP = 75;          % intensity threshold used to turn grayscale into temporary binary image 
                            % b/c the glass slide edge is grayish, not super bright
    SAFETY=250;             % additional cropping margin to make sure glass edge is gone
    
    areaThreshold = 500;    % smaller connected areas are likely dust
    radiCenter = 200;       % defines center disk around centroid
    NLIM=1400;              % minimal number of pixels in blobs to be considered (remove dust/noise)
    NLIMHOLES=1000;         % minimal number of pixels in holes to be considered under large holes
    
    % Auxiliary Variables
    xcirc=cos((0:pi/180:2*pi));  ycirc=sin((0:pi/180:2*pi));
    imageHeight = 512;  
    imageWidth  = 512;
    [x, y] = meshgrid(1:imageWidth, 1:imageHeight);
    qual = 1;   % default quality is 1 meaning assumed to be a good photo/sample
    
    % Determine deepest subfolder name (it contains important info, e.g. salt name)
    pathParts = split(pwd, filesep);
    pathParts = pathParts(~cellfun('isempty', pathParts));
    deepestSubfolder = pathParts{end};
    
    % prepare figure window
    figure(1), set(gcf,'color','w','WindowState','maximized')
    
    % Get list of all PNG images in the current folder
    imageFiles = dir('*.png');
    numImages = length(imageFiles);
    
    for nfn = 1:numImages
        clf
        clear xC yC diskMask numOnesInDisk totalPixelsInDisk compactnessCenter;
    
        % Read the current image
        fn = imageFiles(nfn).name;
        fprintf('Processing image: %s\n', fn);  % Optional: Display file being processed
        imgraw = imread(fn);
        if size(imgraw, 3) == 3
            imgraw = rgb2gray(imgraw);
        end
        imgLow  = imgraw > THRES;           % binary image of all deposit
        imgHigh = imgraw > HIGHTHRES;       % binary image of bright deposit
        
        
        % Detect presence of glas edge and crop
        imgT   = (imgraw>THRESTMP);          % only used for cropping
        colSums = sum(imgT, 1);             % collaps image to a vector
        dum=size(imgT);
        xmin=1; xmax=imageWidth; 
        ymin=1; ymax=imageHeight; 
        
        if max(colSums)>=3000               % vertical edge present 
           indices = find(colSums>=3000);
           if indices(end)<=imageWidth/2    % edge in left half of image
              xmin = indices(end)+SAFETY;
           else                             % edge in right half of image
              xmax = indices(1)-SAFETY;
           end
        end
    
        imgraw  = imgraw(ymin:ymax,xmin:xmax);     % crop for main analysis
        imgLow  = imgLow(ymin:ymax,xmin:xmax);     % crop for main analysis
        imgHigh = imgHigh(ymin:ymax,xmin:xmax);    % crop for main analysis 
    
        % ---------------------------------------------------
        % New analysis medianEccentricity & medianArea ;
    
        connectedComponents = bwconncomp(imgHigh);
        properties = regionprops(connectedComponents, 'Area', 'Eccentricity', 'PixelIdxList');
    
        % Filter components based on area
        largeComponents = properties([properties.Area] >= areaThreshold);
    
        % Display the number of large components found
        disp(['Number of large components: ', num2str(length(largeComponents))]);
    
        % Extract areas and eccentricities from large components
        areas = [largeComponents.Area];
        eccentricities = [largeComponents.Eccentricity];
    
        % Initialize a white background image
        maskedImg = 255 * ones(size(imgraw), 'uint8');  % White background
    
        % Calculate mean and median eccentricity and area
        meanValueAbove50 = mean(imgraw(imgraw > 50));
        count = length(largeComponents);
        meanEccentricity = mean(eccentricities);
        medianEccentricity = median(eccentricities);
        medianArea = median(areas);
    
        % ---------------------------------------------
    
        % New complete metrics by 053124
        
        % -- 1 / old numwhitepixels --
        area = sum(sum(imgLow));
        % -- 2 / number of edge points from binary image (all precipitate) --
        edges = edge(imgLow,'Canny',[0.1 0.3]);
        sumEdgesLow = sum(sum(edges)); 
        clear edges
        % -- 3 / number of edge points from binary image (bright precipitate) --
        edges = edge(imgHigh,'Canny',[0.1 0.3]);
        sumEdgesHigh = sum(sum(edges)); 
        clear edges
        % -- 4 / ratio of all precipitate area over all edges --
        areaOverEdgeLow = area/sumEdgesLow;
        % -- 5 / ratio of bright precipitate area over all edges --
        areaOverEdgeHigh = area/sumEdgesHigh;
        % -- 4 / stddev of all precipitate intensities --
        selectedPixels = imgraw(imgraw > THRES);
        stdRaw = std(double(selectedPixels));
        % -- 5 / total area of bright precipitate --
        selectedPixels = imgraw(imgraw > HIGHTHRES);
        areaHigh = length(selectedPixels);
        % -- 6 / stddev of bright precipitate intensities --
        if areaHigh>=1000
            stdHigh = std(double(selectedPixels));
        else
            stdHigh = -1;
        end
        % -- 7 / median eccentricity of connected regions --    
        connectedComponents = bwconncomp(imgHigh);
        properties = regionprops(connectedComponents, 'Area', 'Eccentricity', 'PixelIdxList');
        largeComponents = properties([properties.Area] >= areaThreshold);
        areas = [largeComponents.Area];
        eccentricities = [largeComponents.Eccentricity];
        medianBrightEccentricity = median(eccentricities);
        % -- 8 / median area of connected regions --
        medianBrightArea = median(areas);
        % -- 9 / compactness of core region --
        [x, y] = meshgrid(1:size(imgLow, 2), 1:size(imgLow, 1));  %-------- defined x,y again --------%
        [rows, cols] = find(imgLow == 1);
        xC = mean(cols);
        yC = mean(rows);
        diskMask = (x - xC).^2 + (y - yC).^2 <= radiCenter^2;
        numOnesInDisk = sum(imgLow(diskMask));
        totalPixelsInDisk = sum(diskMask(:));
        compactnessCenter = numOnesInDisk / totalPixelsInDisk;
        % -- 10&11 / average brightness & fraction of blk pixels of core region --
        brightnessCenter = mean(imgraw(diskMask));
        blackCoreFraction = sum(imgraw(diskMask)<THRES)/sum(imgraw(diskMask));
        % -- 12&13 / skewness and jurtosis of the precipitate intensities --
        intensitiesAboveThres = imgraw(imgraw > THRES);
        intensitySkewness = skewness(double(intensitiesAboveThres));
        intensityKurtosis = kurtosis(double(intensitiesAboveThres));
        % -- 14 / intensity ratio of inner ring-section over outer section --
        [rowsAbove, colsAbove] = find(imgLow);  % Using imgLow which contains thresholded data
        distances = sqrt((colsAbove - xC).^2 + (rowsAbove - yC).^2);
        sortedDistances = sort(distances);
        numPoints = length(sortedDistances);
        rmin = sortedDistances(round(0.1 * numPoints));
        rmax = sortedDistances(round(0.9 * numPoints));
        indicesMid = (distances >= rmin) & (distances <= (rmin + rmax) / 2);
        indicesOuter = (distances > (rmin + rmax) / 2) & (distances <= rmax);
        avgIntensityMid = mean(double(imgraw(sub2ind(size(imgraw), rowsAbove(indicesMid), colsAbove(indicesMid)))));
        avgIntensityOuter = mean(double(imgraw(sub2ind(size(imgraw), rowsAbove(indicesOuter), colsAbove(indicesOuter)))));
        intensityRatio = avgIntensityMid / avgIntensityOuter;
        % -- 15 / angular variations in intensity
        angles = 0:0.25:359.75;         % angles in degrees
        averageIntensities = zeros(size(angles));
        distancesToEdges = [xC, yC, imageWidth - xC, imageHeight - yC];
        dmax = min(distancesToEdges);   % largest search disk size possible
        for i = 1:length(angles)
            angle = angles(i);
            xEnd = xC + dmax * cosd(angle);
            yEnd = yC + dmax * sind(angle);
            [lineX, lineY] = bresenham_line(round(xC), round(yC), round(xEnd), round(yEnd));
            validIndices = (lineX >= 1 & lineX <= size(imgraw, 2)) & (lineY >= 1 & lineY <= size(imgraw, 1));
            lineX = lineX(validIndices);
            lineY = lineY(validIndices);
            if ~isempty(lineX) && ~isempty(lineY)
                indices = sub2ind(size(imgraw), lineY, lineX);
                lineValues = imgraw(indices);
                validValues = lineValues(lineValues > THRES);
                lastValidIndex = find(lineValues > THRES, 1, 'last');
                if ~isempty(validValues)
                    largestRadii(i) = sqrt((lineX(lastValidIndex) - xC)^2 + (lineY(lastValidIndex) - yC)^2);
                    averageIntensities(i) = mean(validValues);
                else
                    largestRadii(i) = 0;
                    averageIntensities(i) = 0;
                end
            else
                largestRadii(i) = 0;
                averageIntensities(i) = 0;
            end
        end
        largestRadii=medfilt1(largestRadii,12);
        stdRays = std(averageIntensities);
        threshold = prctile(averageIntensities, 10);
        lowest10PercentValues = averageIntensities(averageIntensities <= threshold);
        averageLowest10Percent = mean(lowest10PercentValues);
        lowRays = median(averageLowest10Percent);
        stdMaxRays = std(largestRadii);
    
        % -- 15-17 / skeletonization
        skel = bwmorph(imgLow, 'skel', Inf);
        branchPoints = bwmorph(skel, 'branchpoints');
        endPoints = bwmorph(skel, 'endpoints');
        skeletonLength = nnz(skel);  % nnz counts non-zero entries, efficient for binary images
        skeletonBranchPoints = nnz(branchPoints);
        skeletonEndPoints = nnz(endPoints);
        % -- 18 / fractal dimension --
        maxScale = max(size(imgraw));
        scales = round(logspace(log10(1), log10(max(size(imgraw))/10), 20)); % Limit max scale
        uniqueScales = unique(scales);    
        boxCounts = [];
        for scale = uniqueScales
            resizedImage = imresize(imgraw, 1/scale);
            boxCount = sum(sum(resizedImage > THRES));
            boxCounts = [boxCounts; boxCount];
        end
        coeffs = polyfit(log(uniqueScales), log(boxCounts), 1);
        validIndices = boxCounts > 0;
        filteredScales = uniqueScales(validIndices);
        filteredBoxCounts = boxCounts(validIndices);
        if numel(filteredBoxCounts) > 1
            coeffs = polyfit(log(filteredScales), log(filteredBoxCounts), 1);
            fractalDim = -coeffs(1);
        else
            fractalDim(nfn) = NaN;  % Not enough data points to estimate fractal dimension
            display('warning');
        end  
        % -- 19 / entropy --
        glcm = graycomatrix(imgraw);
        stats = graycoprops(glcm);
        contrast = stats.Contrast;
        homogeneity = stats.Homogeneity;
        entropy = -sum(nonzeros(normpdf(double(imgraw(:)))) .* log2(nonzeros(normpdf(double(imgraw(:))))));
        % -- 20 / wavelet analysis --
        [cA,~,~,~] = dwt2(imgraw, 'db1');
        waveletEntropy = -sum(nonzeros(cA(:) / sum(cA(:))) .* log2(nonzeros(cA(:) / sum(cA(:)))));
        %     waveletEnergy(nfn) = sum(cA(:).^2); % correlates strongly with wavewletEntropy
        % -- 21&22 / Gray-Level Co-Occurrence Matrix (GLCM) --
        glcm = graycomatrix(imgraw, 'Offset', [0 1]);  % Horizontal offset
        stats = graycoprops(glcm);
        corrGLCM = stats.Correlation;
        energyGLCM = stats.Energy;
        % -- 23 / local texture via stddev --   
        SE = strel('disk', 5);
        localStd = stdfilt(double(imgraw), SE.getnhood());
        maskedStd = localStd;
        maskedStd(~imgLow) = NaN;
        meanStd5 = nanmean(localStd(:));
        SE = strel('disk', 25);
        localStd = stdfilt(double(imgraw), SE.getnhood());
        maskedStd = localStd;
        maskedStd(~imgLow) = NaN;
        meanStd25 = nanmean(localStd(:));
        SE = strel('disk', 100);
        localStd = stdfilt(double(imgraw), SE.getnhood());
        maskedStd = localStd;
        maskedStd(~imgLow) = NaN;
        meanStd100 = nanmean(localStd(:));
        % -- 24 (this part is slow)
            % C = contours(imgraw, [HIGHTHRES HIGHTHRES]);
            % contourMatrix = C;
            % numC = 0;
            % while ~isempty(contourMatrix)
            %     numPoints = contourMatrix(2, 1);
            %     contourMatrix = contourMatrix(:, numPoints + 2:end);
            %     numC = numC + 1;
            % end
            % numContours = numC;
            % Compute contours
        C = contours(imgraw, [HIGHTHRES HIGHTHRES]);
        numContours = 0;
        if ~isempty(C)
            currentIdx = 1;
            totalCols = size(C, 2);
            
            while currentIdx < totalCols
                % The number of points in the current contour
                numPoints = C(2, currentIdx);
                % Move the index to the next contour (current contour + 1 header + numPoints)
                currentIdx = currentIdx + numPoints + 1;
                % Increment contour count
                numContours = numContours + 1;
            end
        end
        % ====================================================================
        % End of new analysis
        % ====================================================================
    
        % ====================================================================
        % Start old analysis a la PNAS
        % ====================================================================
    
        subplot(2,4,1), imshow(imgraw)
        title('unprocessed gray image');
        axis('equal'),colormap(gray);
        imgT=(imgraw > THRES);
    
        % ================ FIND TOTAL PERIMETER ====================
    
        % Label each connected component in the image
        [L, num] = bwlabel(imgT);
    
        % Extract the area of each labeled region
        stats = regionprops(L, 'Area');
    
        % Find the labels of regions that have an area larger than NLIM
        largeBlobs = find([stats.Area] > NLIM);
    
        % Initialize a new binary image of the same size as imgT with zeros
        newImgT = false(size(imgT));
    
        % Merge the blobs larger than NLIM into the new image
        for k = 1:numel(largeBlobs)
            newImgT = newImgT | (L == largeBlobs(k));
        end
    
        % The areas of the blobs larger than NLIM
        areasOfLargeBlobs = [stats(largeBlobs).Area];
    
        subplot(2,4,2), imshow(newImgT)
    
        % Extract the (y, x) coordinates of the white pixels
        [y, x] = find(newImgT);
        % Compute the mean x and y values
        centroidX = mean(x); centroidY = mean(y);
        hold on, plot(centroidX,centroidY,'g*')
    
        numWhitePixels=sum(sum(newImgT));
    
        % === ellipse saga start ===
    
        % 1. Calculate the central moments
        xbar = mean(x);
        ybar = mean(y);
        u20 = mean((x - xbar).^2);
        u02 = mean((y - ybar).^2);
        u11 = mean((x - xbar).*(y - ybar));
    
        % 2. Construct the covariance matrix
        M = [u20 u11; u11 u02];
    
        % 3. Compute the eigenvalues and eigenvectors of the covariance matrix
        [V, D] = eig(M);
        lambda = diag(D);
    
        % 4. Major and minor axis lengths (proportional)
        a = sqrt(2*lambda(1));
        b = sqrt(2*lambda(2));
    
        % 5. Major-to-minor axis ratio
        axisRatio = max(a,b) / min(a,b);
    
        % 6. Determine the orientation of the ellipse
        if V(1,1) == 0
            if V(2,1) == 0
                theta = 0;
            else
                theta = pi/2;
            end
        else
            theta = atan(V(2,1)/V(1,1));
        end
    
        % 7. Plot the fitted ellipse
        t = linspace(0, 2*pi, 360);
        X_ellipse = xbar + a*cos(t)*cos(theta) - b*sin(t)*sin(theta);
        Y_ellipse = ybar + a*cos(t)*sin(theta) + b*sin(t)*cos(theta);
    
        hold on, plot(X_ellipse, Y_ellipse, 'g', 'LineWidth', 1.25);
        title(sprintf('%d blobs larger than %d pixel\naspect ratio = %.4f',numel(largeBlobs),NLIM,axisRatio))
    
        % === ellipse saga end ===
    
    
        subplot(2,4,3), imshow(imgT-newImgT), title('bright data being ignored')
    
        % Combine all the large blobs into a single binary image
        combinedLargeBlobs = false(size(newImgT));
        for k = 1:numel(largeBlobs)
            combinedLargeBlobs = combinedLargeBlobs | (L == largeBlobs(k));
        end
    
        % Compute the outer perimeter of the combined large blobs
        perimCombined = bwperim(combinedLargeBlobs);
    
        % Extract the (x, y) coordinates of the perimeter pixels
        [y, x] = find(perimCombined);  % Note: 'find' gives (row, column), i.e., (y, x)
    
        % Display newImgT with the outer perimeter of the large blobs
        subplot(2,4,4), imshow(newImgT), title('Large Blobs with Outer Perimeter');
        hold on;
        plot(x, y, 'r.', 'MarkerSize', 1);  % Overlay the perimeter with red dots
    
        % Combine all the large blobs into a single binary image
        combinedLargeBlobs = false(size(newImgT));
        for k = 1:numel(largeBlobs)
            combinedLargeBlobs = combinedLargeBlobs | (L == largeBlobs(k));
        end
    
        % Fill the holes in the combined image
        filledBlobs = imfill(combinedLargeBlobs, 'holes');
    
        % Compute the outer perimeter of the filled large blobs
        perimCombined = bwperim(filledBlobs);
        perimeterLength = sum(perimCombined(:));
    
        % Extract the (x, y) coordinates of the perimeter pixels
        [y, x] = find(perimCombined);  % Note: 'find' gives (row, column), i.e., (y, x)
    
        % Display newImgT with the outer perimeter of the large blobs
        subplot(2,4,4), imshow(newImgT), hold on
        plot(x, y, 'r.', 'MarkerSize', 1);  % Overlay the perimeter with red dots
        title(sprintf('total outer perimeter: %d pixel\n', perimeterLength))
    
        % ===== ANALYZE HOLES ==========
    
        % 1. Identify holes in newImgT by subtracting it from its filled version
        holes = imfill(newImgT, 'holes') - newImgT;
    
        % 2. Label each hole
        [L_holes, num_holes] = bwlabel(holes);
    
        % 3. Extract the area of each hole
        stats_holes = regionprops(L_holes, 'Area');
    
        % Identify holes that are larger than NLIMHOLES pixels
        largeHoleIndices = find([stats_holes.Area] > NLIMHOLES);
        largeHoleAreas = [stats_holes(largeHoleIndices).Area];
    
        % Count of large holes
        countLargeHoles = numel(largeHoleIndices);
    
        % Convert holes to RGB with white holes
        mergedHolesImage = uint8(cat(3, holes, holes, holes) * 255);
    
        % Paint large holes in red
        for idx = 1:countLargeHoles
            currentLargeHole = (L_holes == largeHoleIndices(idx));
            mergedHolesImage(:,:,1) = mergedHolesImage(:,:,1) + uint8(currentLargeHole * 255); % Set red channel
            mergedHolesImage(:,:,2) = mergedHolesImage(:,:,2) - uint8(currentLargeHole * 255); % Clear green channel
            mergedHolesImage(:,:,3) = mergedHolesImage(:,:,3) - uint8(currentLargeHole * 255); % Clear blue channel
        end
    
        % Displaying results
        subplot(2,4,5), imshow(mergedHolesImage)
        title(sprintf('%d holes larger than %d pixels\nmedian large hole area is %.0f pixel\nlargest hole area is %.0f pixel',countLargeHoles,NLIMHOLES,median(largeHoleAreas),max(largeHoleAreas)));
    
        numBlackPixels=sum(sum(largeHoleAreas));
    
    
        % ====== FIND DISTANCES TO CENTROID ======
    
        % Calculate distances from the centroid to each white pixel in newImgT
        distances = sqrt((x - centroidX).^2 + (y - centroidY).^2);
    
        % Plot histogram of distances in subplot 6
        subplot(2,4,6)
        h = histogram(distances, 50); % Adjust the second parameter (e.g., 100) to change the number of bins
        xlabel('centroid distance (pixel)');
        ylabel('white pixel count');
        axis('square')
    
        % Determine the most frequently occurring distance (mode)
        [~, modeIdx] = max(h.Values);
        modeDistance = (h.BinEdges(modeIdx) + h.BinEdges(modeIdx+1))/2; % Middle of the bin
    
        % Indicate the mode distance on the histogram
        hold on;
        plot([modeDistance, modeDistance], ylim, 'r-', 'LineWidth', 1.5);
        title(sprintf('distances to centroid\nmean=%.1f, std=%.1f pixel\nmode=%.1f, median=%.1f pixel',mean(distances), std(distances),modeDistance,median(distances)));
    
    
    
        % ======== WHITE IMAGE EROSION WITH SMALL DISKS ==========
        i=0;
        tmp=1;
        while tmp>=0.1;
            i=i+1;
            tmp=(sum(sum(imerode(newImgT,strel('disk', i-1)))))/sum(sum(newImgT));
            frct(i)=tmp;
        end
        p = polyfit((1:5)-1, frct(1:5), 1); 
        % subplot(2,4,7), plot((1:i)-1,frct(1:i),'bo',(0:4),p(1)*(0:4)+p(2),'b-')
        % axis('square'),xlabel('disk radius'),ylabel('fraction of white pixels')
        % % p(1) is the slope and p(2) is the y-intercept
        erosionslope = -p(1);
        frct01 = i-1;
        % title(sprintf('erosion of white pixels with disks\ninitial slope=%.5f\nf=0.1 for disk size=%d',erosionslope,frct01));
    
    
        subplot(2,4,7) 
        imagesc(imgraw); colormap(gray); axis image; hold on
        plot(xC,yC,'g*')
        plot(xC+radiCenter*xcirc,yC+radiCenter*ycirc,'b-')
        title(nfn)
    
        % prepare out of metric vector
        p1=numWhitePixels;
        p2=numBlackPixels;
        p3=numBlackPixels/numWhitePixels;
        p4=numel(largeBlobs);
        p5=perimeterLength;
        p6=axisRatio;
        p7=countLargeHoles;
        p8=median(largeHoleAreas);  % no hole case is handled as zero area
        if isnan(p8)
            p8 = 0;
        end
        if isempty(p8)
            p8 = 0;
        end
        p9=max(largeHoleAreas);     % no hole case is handled as zero area
        if isnan(p9)
            p9 = 0;
        end
        if isempty(p9)
            p9 = 0;
        end
    
        p10=mean(distances);
        p11=std(distances);
        p12=modeDistance;
        p13=median(distances);
        p14=skewness(distances);
        p15=erosionslope;
        p16=frct01;
        p17=medianEccentricity;
        if isempty(largeComponents)
            p17 = 0;
        end
        p18=medianArea;
        if isempty(largeComponents)
            p18 = 0;
        end
        p19=sumEdgesLow./area;               % !!!!
        p20=sumEdgesHigh./area;              % !!!!
        p21=areaOverEdgeLow;
        p22=areaOverEdgeHigh;
        if isempty(p20)
            p22 = 0;
        end
        p23=stdRaw;
        p24=areaHigh;
        p25=stdHigh;
        p26=compactnessCenter;
        p27=brightnessCenter;
        p28=blackCoreFraction;
        p29=intensityKurtosis;
        p30=intensitySkewness;
        p31=intensityRatio;
        p32=skeletonLength/area;           % !!!!
        p33=skeletonBranchPoints/area;     % !!!!
        p34=skeletonEndPoints/area;        % !!!!
        p35=fractalDim;                     
        p36=log10(entropy)/area;           % !!!!
        p37=waveletEntropy/area;           % !!!!
        p38=stdRays;
        p39=lowRays;
        p40=stdMaxRays;
        p41=corrGLCM;
        p42=energyGLCM;
        p43=meanStd5/area;                 % !!!!
        p44=meanStd25/area;                % !!!!
        p45=meanStd25/meanStd5;            % !!!!
        p46=meanStd100/meanStd25;          % !!!!
        p47=numContours/area;              % !!!!
    
        % ======== DISPLAY NUMERIC RESULTS =============
    
        % Assuming your variables are already defined, you can proceed with the following:
    
        subplot(2,4,8);
        axis off;  % Turn off axis since this is a display of variable values.
    
        fn_disp = strrep(fn, '_', '\_');
        % Create a text string with all the variable values:
        textString = { ...
            ['fn: ', fn_disp], ...
            ['THRES: ', num2str(THRES)], ...
            ['NLIM: ', num2str(NLIM)], ...
            ['NLIMHOLES: ', num2str(NLIMHOLES)], ...
            ['numWhitePixels: ', num2str( p1 )], ...
            ['numBlackPixels: ', num2str( p2 )], ...
            ['ratio: ', num2str( p3 )], ...
            ['numLargeBlobs: ', num2str( p4 )], ...
            ['perimeterLength: ', num2str( p5 )], ...
            ['axisRatio: ', num2str( p6 )], ...
            ['countLargeHoles: ', num2str( p7 )], ...
            ['medianLargeHoleAreas: ', num2str( p8 )], ...
            ['maxLargeHoleAreas: ', num2str( p9 )], ...
            ['meanDistances: ', num2str( p10 )], ...
            ['stdDistances: ', num2str( p11 )], ...
            ['modeDistances: ', num2str( p12 )], ...
            ['medianDistances: ', num2str( p13 )], ...
            ['skewnessDistances: ', num2str( p14 )], ...
            ['erosionslope: ', num2str( p15 )], ...
            ['frct01: ', num2str( p16 )], ...
            ['medianEccentricity : ', num2str( p17 )], ...
            ['medianArea: ', num2str( p18 )], ...
            ['sumEdgesLow: ', num2str( p19 )], ...
            ['sumEdgesHigh: ', num2str( p20 )], ...
            ['areaOverEdgeLow: ', num2str( p21 )], ...
            ['areaOverEdgeHigh: ', num2str( p22 )], ...
            ['stdRaw: ', num2str( p23 )], ...
            ['areaHigh: ', num2str( p24 )], ...
            ['stdHigh: ', num2str( p25 )], ...
            ['compactnessCenter: ', num2str( p26 )], ...
            ['brightnessCenter: ', num2str( p27 )], ...
            ['blackCoreFraction: ', num2str( p28 )], ...
            ['intensityKurtosis: ', num2str( p29 )], ...
            ['intensitySkewness: ', num2str( p30 )], ...
            ['intensityRatio: ', num2str( p31 )], ...
            ['skeletonLength: ', num2str( p32 )], ...
            ['skeletonBranchPoints: ', num2str( p33 )], ...
            ['skeletonEndPoints: ', num2str( p34 )], ...
            ['fractalDim: ', num2str( p35 )], ...
            ['log10Entropy: ', num2str( p36 )], ...
            ['waveletEntropy: ', num2str( p37 )], ...
            ['stdRays: ', num2str( p38 )], ...
            ['lowRays: ', num2str( p39 )], ...
            ['stdMaxRays: ', num2str( p40 )], ...
            ['corrGLCM: ', num2str( p41 )], ...
            ['energyGLCM: ', num2str( p42 )], ...
            ['meanStd5: ', num2str( p43 )], ...
            ['meanStd25: ', num2str( p44 )], ...
            ['ms25over5: ', num2str( p45 )], ...
            ['ms100over25: ', num2str( p46 )], ...
            ['numContours: ', num2str( p47 )] ...
        };
    
    
    
        % Display the text on the subplot:
        text(0, 1, textString, 'FontSize', 4, 'VerticalAlignment', 'top');
    
        % === FILE OUTPUT === 
        pause(0.5)
    
        % Remove the file extension and prepend "results_"
        [~, nameOnly, ~] = fileparts(fn);
        jpg_filename = ['results_' nameOnly '.jpg'];
    
        % Save the active figure as a .jpg
        saveas(gcf, jpg_filename, 'jpg');
    
    
        % Check if file exists and is empty
        if ~exist(filename, 'file') || isempty(dir(filename).bytes)
            writeHeader = true;
        else
            writeHeader = false;
        end
        % Open file in append mode
        fid = fopen(filename, 'a');
        % Write header if necessary
        if writeHeader
            fprintf(fid, 'directory\tfn\tQuality\tTHRES\tHIGHTHRES\tNLIM\tNLIMHOLES\tnumWhitePixels\tnumBlackPixels\tratio\tnumLargeBlobs\tperimeterLength\taxisRatio\tcountLargeHoles\tmedianLargeHoleAreas\tmaxLargeHoleAreas\tmeanDistances\tstdDistances\tmodeDistances\tmedianDistances\tskewnessDistances\terosionslope\tfrct01\tmedianEccentricity\tmedianArea\tsumEdgesLow\tsumEdgesHigh\tareaOverEdgeLow\tareaOverEdgeHigh\tstdRaw\tareaHigh\tstdHigh\tcompactnessCenter\tbrightnessCenter\tblackCoreFraction\tintensityKurtosis\tintensitySkewness\tintensityRatio\tskeletonLength\tskeletonBranchPoints\tskeletonEndPoints\tfractalDim\tlog10Entropy\twaveletEntropy\tstdRays\tlowRays\tstdMaxRays\tcorrGLCM\tenergyGLCM\tmeanStd5\tmeanStd25\tms25over5\tms100over25\tnumContours\n');
        end
    
        % Write data
        fprintf(fid, '%s\t%s\t%1d\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n',deepestSubfolder,fn, qual, THRES, HIGHTHRES, NLIM, NLIMHOLES, p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47);
    
        % Close file
        fclose(fid);
    
    end     % <-- closes fnf loop

end



% ======= AUXILIARY FUNCTION =======
function [x, y] = bresenham_line(x1, y1, x2, y2)  % for diagonal lines on grid
    % Initialize output coordinate arrays
    x = [];
    y = [];
    
    % Calculate differences and steps
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    sx = sign(x2 - x1);
    sy = sign(y2 - y1);
    
    % Initialize error term
    err = dx - dy;  % This should consider both dimensions initially.
    
    while true
        x = [x, x1];
        y = [y, y1];
        if x1 == x2 && y1 == y2
            break;
        end
        e2 = 2 * err;  % Multiply by 2 to avoid using fractions.
        if e2 > -dy  % Adjust x if the error is greater than -2*dy
            err = err - dy;
            x1 = x1 + sx;
        end
        if e2 < dx   % Adjust y if the error is less than 2*dx
            err = err + dx;
            y1 = y1 + sy;
        end
    end
end

