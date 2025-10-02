SyntheticImageGeneration_Procedure.txt - This file provides instructions for training a LoRA-fine-tuned Stable Diffusion model and generating synthetic images of salt crystal deposits. 

requirements.txt – Lists all Python dependencies needed for LoRA fine-tuning and inference with Stable Diffusion, including PyTorch, diffusers, and related libraries.

inference_generate_images.py – Loads a LoRA-fine-tuned Stable Diffusion model and generates multiple synthetic images of salt crystals with reproducible random seeds.


MATLAB Scripts:

1.	removeOutliers.m - Reads all .png images from an input folder, removes those that are too dark, too colorful, or have bright/noisy borders, and then selects exactly 676 images. Copies the filtered set into an output folder and verifies the count.

2.	ResizeSynImage.m - Contains the function resizeSynImg which rescales a grayscale salt-stain image so that its number of white pixels matches a given target. If the target is zero, it outputs a blank canvas. The function preserves the background tone, resizes with a random “fudge factor” for variability, and then centers or crops the result to the original image size.

3.	ResizeSynImage_Function.m - Defines the function resizeSynImg, which rescales a grayscale salt-stain image so that its white-pixel count matches a specified target (NUMWHTPIXTARGET) using a threshold (THRES). If the target is ≤ 0, the function outputs a blank canvas. The scaling factor is adjusted with a small random “fudge factor,” and the resized image is centered or cropped to the original dimensions before saving.

4.	Normalise_Intensity.m - Script that rescales synthetic images so their intensity range matches experimental reference images. It computes “low” and “high” anchors (border medians and brightest pixels) from the experimental set, then linearly rescales each synthetic image to align with those anchors. Includes helper functions anchorMedians and anchorForOne for per-image and set-level statistics.

5.	Main_Image_Processing.m - Script that analyzes raw images to extract morphological, intensity, and texture metrics. It identifies large blobs and holes, computes perimeters, distances from centroids, skeleton and fractal properties,  performs erosions, and visualizes key features with overlays and histograms.

6.	Zscore_CombinedData.m - Script that combines experimental and synthetic image-derived metrics, selecting up to 128 synthetic images per salt, and performs global z-scoring on the metric columns. It outputs a combined, standardized table and provides an optional heatmap visualization of the z-scored metrics.
	
7.	MLP_TrainingValidationAccuracy.m - Script that trains multilayer perceptron (MLP) models on real, synthetic, and combined datasets of image-derived metrics. It evaluates test accuracy for predicting the salt type across varying experimental training 	images N, records per-epoch training and validation accuracies, and generates plots showing mean ±1σ performance trends.
	
8.	RF_TrainingValidationAccuracy.m - Script that trains Random Forest models on real, synthetic, and combined datasets of image-derived metrics. It evaluates test accuracy for predicting salt type across varying experimental training images N, records training and out-of-bag (OOB) validation accuracy across trees, and generates plots showing mean ±1σ performance trends.
	
9. 	XGBoost_TrainingValidationAccuracy.m - This script trains XGBoost models on real, synthetic, and combined datasets of image-derived metrics to predict salt type. It computes test accuracy over multiple repetitions, tracks training and validation accuracy across boosting rounds, and plots performance with mean ±1σ trends for each dataset type.
	
10.	MLP_Predict_Expt_or_Syn_ts128.m - This script trains a multilayer perceptron (MLP) to classify samples as either experimental or synthetic using image-derived metrics. It averages results over multiple training runs, predicts the type for held-out test data, calculates mean test accuracy, and plots a row-normalized confusion matrix showing class-specific prediction performance.
	
11.	RF_Predict_Expt_or_Syn_ts128.m - This script trains a random forest (RF) classifier to distinguish experimental from synthetic samples using image-derived metrics. It averages results over multiple runs, predicts the type for held-out test data, calculates mean test accuracy, and displays a row-normalized confusion matrix for class-specific performance.
	
12.	XGB_Predict_Expt_or_Syn_ts128.m - This script trains an XGBoost classifier to distinguish experimental from synthetic samples using image-derived metrics. It averages results over multiple runs, predicts the type for held-out test data, calculates mean test accuracy, and displays a row-normalized confusion matrix for class-specific performance.
	
13.	ResNet50.m - This script trains a ResNet-50 convolutional neural network to classify salts from image datasets of varying sizes (synthetic, experimental, and combined). It loops over multiple training set sizes and replicates, constructs appropriate image datastores, fine-tunes the network’s final fully connected layer, and evaluates performance on a fixed test set. The script records test accuracy, training/validation accuracy, and loss for each condition, and plots test accuracy versus training size as well as average training-validation curves for both accuracy and loss.

14.	PCA_2d.m - This script performs principal component analysis (PCA) on a reference experimental dataset (ExptData128) and projects multiple other datasets (e.g., ts2, ts8, ts64) onto the same PCA space. It z-scores the features, handles missing values, and visualizes 2D projections (PC1 vs. PC2) of all datasets with consistent axis limits, color-coded by salt type. The final figure includes a subplot of the PCA training data and labeled projections for each dataset size.

	
