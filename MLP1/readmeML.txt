###########################
Machine Learning Project 1
###########################
 
### Team: Ohlala

### Members:
- Barthelemy Bidot
- Yannick Wolff
- Lionel Trebuchon

### Summary

Here is a summary to explain the work done to solve this project:

1) Perform preprocessing

We have had rather little pre-processing efforts on our images, as most of our features necessitated all of the voxels. However, we have taken basic steps of preprocessing, such as:
- cutting out black pixels which would otherwise obliviate counting techniques and do not add any information
- cutting out light pixels that seemed to be inherently random. This step is more questionnable, because a lasso algorithm could take care of that. However, we optimized our preprocessing for a Ridge regression. Graphs of the frequency of the voxels intensities made the following prior non-negligible: voxels above 1400 are randomly distributed and do not add any information to the regression. A limit of 2400 was chosen as consensus so as to not disable features not recognizable by the human eye.
- a 2D implementation of the Fourier transform has also been tried as preprocessing. However, the features resulting from it (primary frequencies count and shape patterns) have not been found to be conclusive based on submissions we made and our own judgment. This step is also presented in the feature engineering (see 2).


2) Extracting features

The main work of the project has consisted in extracting appropriate features. We tried 5 different approaches to get the best features. 3 of them were not conclusive:
- we analyzed the brain volume to check for any evolution over the age. All the volumes were the same (the images were probably normalized already) so we couldn't extract any information from this.
- we performed a Fourier analysis to assess the variance of the intensity. The idea was to find a way to compute the "roughness" of the white matter which we visually thought to increase with the age. Unfortunately, after qualitatively assessing the results for each image, we didn't find any relevant patterns.
- we tried a similar approach by derivating the pixels according to a given axis in order to analysis the variance but the results were not significant.

The remaining 2 approaches gave us relevant features that were used for the prediction:
- we performed a global histogram for each image (numbers of pixels with respect to ranges of intensities). When plotting the results, we noticed the existence of 2 to 3 main peaks for which the intensity and number of pixels were changing over the age. The first peak, which does not always exist for all patients, can be seen graphically on the given images as a white "cross" that becomes darker over the age. We haven't had time to broaden our analysis on this aspect but we think that a strong feature can be found there. We focused our analysis on the 2 other peaks. We used another algorithm to extract their intensity and frequency and obtained thus a set of 4 features per image that we used in our regression.
- we also performed histograms locally. We divided each image into 48 smaller subsets (more precisely, in 3*16 flat cuboids, or sections, as we considered 16 surfaces per axis). For each surface, we computed 4 histograms with different bins (respectively 16, 32, 48 and 64 bins). We used the frequencies of these histograms to obtain 4 sets of features per image (each set containing 48*number_of_bins features) that we then used in our regression. After some research online, we were able to confirm that these features were representing in particular and to a certain extent, the amount of white and grey matter in the brain (which change over the age).

3) Train model
For more clarity, let's call the set of 4 features related to the peaks, set 1, and the set of features related to the local histograms, set 2 (regardless of the number of bins chosen).
We implemented 3 regressions : linear, Ridge and Lasso regressions (with regularization parameter alpha).
For the set 1, we used the 3 regressions over a large range of alpha. We chose the alpha which gave the best score and both the linear and Ridge regressions gave the same results while Lasso, by lack of features, was not efficient. This was our first submission which was not very good as we were lacking features.
For the set 2, we applied the same process and obtained an optimized result with Ridge, alpha = 1 and 48 bins.
Finally, we tried with the set 1 and 2 which makes for 4 + 48*number_of_bins features. We obtained our best score, which is our current submission, for Ridge, alpha = 1 and 48 bins.

Remarks: i)  We tried Ridge and Lasso with the set 1 and the nearly full set 2 (bins 32, 48 and 64 together), i.e. with 4 + 48*(32+48+64) features. However, we didn't obtain good results.
	ii) We also implemented cross-validation with 10 folders as we were strongly aware that our model validation was being done on the same data as the one used for the training, which was not good (overfitting). Unfortunately, without understanding why, this cross-validation tended constantly to increase alpha to impossible values (10^9 for example). This was not consistent and was giving us wrong results. Thus, we didn't use it for our submissions.

4) Perform postprocessing
We have not performed many postprocessing on our results. To find the ages in an integer fashion, we rounded down the results (considering a patient gains one year the day of his birth).

### Files: there are in the archive 2 python scripts to generate the features, 7 csv files with these features already generated, and the file predict_final.py which does the prediction and create the file prediction.csv which contains our submission.

More precisely, there are:
- The pyhton script to compute the 4 peaks features: peaksFeatures.py
- It generates 6 csv features files, 1 for each coordinate of each peak: p1_x.csv (not used), p1_y.csv (not used), p2_x.csv, p2_y.csv, p3_x.csv, p3_y.csv
- The python script to compute the 3*16*48 section features: sectionFeatures.py
- It generates 1 csv features file, sectionFeatures.csv, with all the features of each images (1 line is 1 image)
- The python script predict_final.py, which concatenate the features, operate the regression, predict the values and save them in the file prediction.csv.
