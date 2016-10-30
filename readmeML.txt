Readme: It is required to include a readme file which contains at least the team's name, the (real) names of the authors and a summary of the approach used for the final submission. The summary should be a short list like this:
1) Perform preprocessing W
2) Select k voxels according to X
3) Train model Y
4) Perform postprocessing Z
Any other comments on why you do the things you do are strongly encouraged.
Code: Include one file "predict_final.py" (or .m if Matlab). When this script is run, it produces a .csv file in the correct submission format which contains your final submission. (if your algorithm contains non-deterministic elements it is ok when the output changes slightly compared to the one you submitted online). The structure of the remaining code is up to you.
Commenting your code is strongly encouraged as well.

*******************************

Readme

Team : Ohlala
Members :
- Barthelemy Bidot
- Yannick Wolff
- Lionel Trebuchon

Summary
Here is a short summary to explain the work done to solve this project.

1) Perform preprocessing
We have had rather little pre-processing efforts on our images, as most of our features necessitated all of the voxels. However, we have taken basic steps of preprocessing, such as:
- cutting out black pixels which would otherwise obliviate counting techniques and do not add any information
- cutting out light pixels that seemed to be inherently random. This step is more questionnable, because a lasso algorithm could take care of that. However, we optimized our preprocessing for a ridge regression. Graphs of the frequency of the voxels intensities made the following prior non-negligible: voxels above 1400 are randomly distributed and do not add any information to the regression.
- a 2D implementation of the fourier transform has also been tried as preprocessing. However, the features resulting from it (primary frequencies couting, and shape patterns) has not been found to be conclusive based on the admissions. This step also present in the feature engineering (see 2).

2) Select k voxels according to X
The main work of the project has consisted in extracting appropriate features. We tried 5 different approaches to get the best features. 3 of them were not conclusive:
1) we analyzed the brain volume to check for any evolution over the age. All the volumes were the same (the images were probably normalized already) so we couldn't extract any information from this.
2) we performed a Fourier analysis to assess the variance of the intensity. Unfortunately, after assessing qualitatively the results for each image, we didn't find any relevant pattern.
3) we tried a similar approach by derivating the pixels according to a given axis in order to analysis the variance but the results were not significant.
The remaining 2 approaches gave us relevant features that were used for the prediction:
4) we performed a global histogram for each image. When plotting the results (after removing the zero values), we noticed the existence of 2 to 3 main peaks for which the intensity and frequency was changing over the age. The first peak, which does not always exist, can be seen graphically on each image as a white "cross" that goes darker over the age. We haven't had time to broaden our analysis on this aspect but we think that a strong feature can be found there. We focused our analysis on the 2 other peaks. We used another algorithm to extract their intensity and frequency and obtained thus a set of 4 features per image that we used in our regression.
5) we performed again histograms but this time locally. For each image, we divided it in 48 smaller 2D-surfaces (16 surfaces per axis). For each surface, we computed 3 histograms with different bins (respectively 32, 48 and 64 bins). We used the frequencies of these histograms to obtain 3 set of features per image (each set containing 16*3*number_of_bins features) that we then used in our regression. After some research online, we were able to confirm that these features were representing in particular and to a certain extent, the amount of white and grey matter in the brain (which change over the age).

3) Train model Y
For more clarity, let's call the set of 4 features related to the peaks, set 1, and the set of features related to the loca histograms (in fact there is 3 subsets there), set 2.
We implemented 3 regressions : linear, Ridge and Lasso regressions (with regularization parameter alpha).
For the set 1, we used the 3 regressions over a large range of alpha. We chose the alpha which gave the best score and both the Ridge and the linear regressions gave the same results while Lasso, by lack of features, was not efficient. This was our first submission which was not very good as we were lacking features.
For the set 2 (without the set 1), we applied the same process and obtained an optimized result with Ridge, alpha = 1 and 48 bins.
Finally, we tried with the set 1 and 2 which makes for 4 + 16*3*number_of_bins features. We obtained our best score, which is our current submission, for Ridge, alpha = X and 48 bins.

Remarq : i)  We tried Lasso with the full set 1 and 2, i.e. with 4 + 16*3*(32+48+64) features. However, we got worse results.
	 ii) We also implemented cross-validation with 10 folders as we were strongly aware that our model assessment was being done on the same data as the one used for the training, which was not really good (overfitting). 	     Unfortunately, without understanding why, this cross-validation tended constantly to increase alpha, to impossible values (10^9 for example). This was not consistent and was thus giving wrong results.

4) Perform postprocessing Z
We have not performed much postprocessing on our results. To find the ages in an integer fashion, we were forced to use a nearest rounding.
A PART L ARRONDI, JE VOIS PAS DE QUOI PARLER ICI.

Code
Get the features
	- 1 code for the 4 peaks
	- 1 code for the 48 range features
Concatenate the features, operate the regression, optimise the alpha, predict the values and save them in the csv file properly
- predict_final.py