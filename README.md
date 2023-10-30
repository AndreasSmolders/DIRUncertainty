# DIRUncertainty
Code regarding publication ["Deep learning based uncertainty prediction of deformable image registration for contour propagation and dose accumulation in online adaptive radiotherapy"](https://doi.org/10.1088/1361-6560/ad0282) by Smolders A, Lomax AJ, Weber DC and Albertini F. Please refer to this publication in case of use. For academic purposes only. In case of commercial interest please contact the corresponding author.

This repository contains code to predict the uncertainty of a given deformable vector field. Up until now, the code was designed for CT-to-CT registration, and will likely not work for other registrations. CT-to-syntheticCT should also work, but it was never tested. The codebase contains:
- Readers for various imaging formats (dicom,mha,nrrd,nifti,...) to read the input images (fixed and moving images) and corresponding vector fields. For vector fields, it is better to use another format than DICOMr. The DICOM standard for vector fields is difficult to read, and we did not succeed in making a general reader. Therefore try to convert them yourself to a simpler format, or make an issue in this repo so we try to read them.
- A pipeline which predicts the uncertainty associated with the registration. This registration is based on the two images (i.e. the local contrast) and the DVF (i.e the registration itself)
- Writers to write the predicted uncertainty into any of the above file formats
- Three follow up processing pipelines:
  1. A class which uses the probabilistic DVF to warp a set of contours. The class can either return individual samples, or it can run a number of samples (e.g. 100) and convert these into voxel-wise probablities for each contour. The input is a structureset in Dicom (RTSTRUCT), but by digging deeper into the code base also other inputs could be used.
  2. A class which uses the probabilistic DVF to warp a dose. Similary to the contours, it creates warped dose samples, which can be either used individually or assembled to get voxel-wise dose uncertainties.
  3. A class which uses a list of probabilistic DVFs and doses to probablistically accumulate dose on a reference scan

## Installation
1. Clone this repository on your local system
2. Install a python > python 3.6
3. Install the requirements 
```
$ pip install -r requirements.txt
```
4. Download the model zoo with pretrained model weights
```
$ wget -O TrainedModels.zip https://www.dropbox.com/s/aa1858x03wpz2e8/TrainedModels.zip?dl=0
$ unzip TrainedModels.zip
```

## Usage
Run the runner.py script. In there, you can fill in the file paths to the fixed and moving images and also the DVF. Also give the path to the appropriate weights of the model in from the model zoo. Depending on your application, you can either save the results or run also the contour propagation or dose warping.

IMPORTANT NOTE: 

- For contour propagation: the **moving** image is the image on which you have the contours, the **fixed** image is the one towards which you warp the contours.
- For dose warping: the **moving** image is the image on which you have the dose, the **fixed** image is the one towards which you warp the dose.

## Validation

Depending on the anatomical site or the quality of the DIR algorithm used in the registration, the model should predict different uncertainties. Similary to normal DIR algorithms, our models contain hyperparameters, which affect the predicted uncertainty and therefore the approriate hyperparameters for each site/DIR need to be selected. In the model zoo, a set of pretrained models are available for a variety of hyperparameters. Most refer to the unsupervised model, and their name reflects their hyperparameters (e.g. L5-1S1-3 corresponds to Lambda=0.5 and Sigma=10^-3). Also the supervised model is publicly available, but this will not be very useful on its own. It is however useful to combine it with an unsupervised model, as was done in the paper. There is also a pipeline available which does this automatically for you.

To select an appropriate model, you can look at the published paper. In case you have a set of scans with landmarks in the fixed and moved images, this would be the preferred validation method. Since landmarks are not readily available, the validation can also be done using contours, if you have a set of fixed and moving images with corresponding manually delineated contours.
### Landmarks

1. Run your DIR algorithm for the set of images of the anatomical region under study. Save the DVFs somewhere on your disk. It is important to note that our models assume that the images are (approximately rigidly registered). Small rigid misalignment are likely to be fine. However, if the images are from two different frame of references, the models will likely fail as the vector field will contain very large values. Therefore, first rigidly align the images before running the deformable part.
2. Run the uncertainty prediction for each image pair with different models from the model zoo. In general, the lower lambda, the higher the predicted uncertainty, and the lower sigma, the larger the difference in uncertainty between region with and without contrast. However, both also influence each other so it is best to run it for several combinations of both hyperparameters. 
3. For each landmark, calculate the error in the x,y and z directlion **individually**. Since the uncertainty prediction is independent for each direction, we treat also the error independently in each direction.
4. Interpolate for each landmark location in the fixed image the predicted uncertainty map. Now you have for each landmark and direction a predicted uncertainty and a registration error. (From now on when referring to landmarks, we actually mean each direction of each landmark)
5. Group together landmarks with similar predicted uncertainty. We took bins of 0.25 mm, and we had 3000 landmarks with each 3 directions (i.e. 9000 data points). For each group, calculate the root-mean-squared-error. 
6. Plot the mean of each bin versus the rmse for various model. The best model is the one that has the RMSE similar to the prediction uncertainty for a wide range of uncertainties. Furthermore, it would be preferred to have as much as possible range, i.e. a model that always predicts an uncertainty of 3 mm is less good than one that sometimes predicts 1mm and sometimes 6mm, even though both types of predictions might fit the data well.

### Contours

1. Run your DIR algorithm for the set of images of the anatomical region under study. Save the DVFs somewhere on your disk. It is important to note that our models assume that the images are (approximately rigidly registered). Small rigid misalignment are likely to be fine. However, if the images are from two different frame of references, the models will likely fail as the vector field will contain very large values. Therefore, first rigidly align the images before running the deformable part.
2. Run the uncertainty prediction for each image pair with different models from the model zoo. In general, the lower lambda, the higher the predicted uncertainty, and the lower sigma, the larger the difference in uncertainty between region with and without contrast. However, both also influence each other so it is best to run it for several combinations of both hyperparameters. 
3. For scan pair, take propagate the contours probabilistically with the method provided above. We used 100 samples, to save time, but more samples would also be good. 
4. Now you have a set of scans with for each contour a probability in each voxel that that voxel is inside the actual contour. Similarly to the landmarks, group all voxels with similar probability together in one group, and calculate the actual fraction of those voxels that was inside the manual contour delineated by the physician. You can either do this for all contours together, or repeat it for each individually (e.g. only for the CTV, only for Esophagus,...). 
5. Plot the average probabiltity of each group versus the actual fraction of voxels inside the contour. Ideally, these are equal for a wide range of probabilities. The fit can also be quantified in one single value, the expected calibration error (ECE) (google it :) ). We discarded all voxels with a p>0.99 or p<0.01, as these are voxels far away from the boundaries of the contour and are therefore very easy to be correct. Including these voxels makes the ECE very small, as there are so many of them, but this is all irrelevant and therefore is excluded from the analysis.
6. The best model is the one with the lowest ECE. However, you might care less about the voxels with probability above 95% or below 5%, as these are easy to predict. Therefore it is important to look at the reliability diagram itself. It can be interpreted as follows:
   - You expect the point (0.5,0.5) to be on your curve, if your DIR algorithm is doing a good job on average, i.e it is not systematically over or undersegmenting the contours.
   - If your line is above the identity line below 0.5 and below the identity line above 0.5, i.e. if your line is more horizontal than the identity line, your model is overconfident. This means that you have to increase the magintude of the uncertainty prediction, e.g. by lowering lambda. If the inverse is true, your model is underconfident.
   - At the moment, we do not direcly see how you can relate the tradeoff between contrast and uncertainty directly to the reliability diagam of the contours. You would have to overlay the probabilities with the image, to see if the gradient of the probabilities is large in regions with low contrast and small in regions with high contrast. For instance, if you see that the gradient is small around a bony area, it probably means that the uncertainty is too large there as the bones give a lot of information for the contouring and therefore would imply a sharp gradient of the probablities, i.e. a very well defined contour boundary. In that case, you would have to lower sigma so that the trade off between contrast and uncertainty is more sensitive. 

The interpretation of the validation is not straightforward, especially for the contours. In doubt, feel free to reach out to us, we are happy to assist you in the process, both with the implementation and interpretation. Either raise an issue here, or contact us (author contact information is included in the paper).
