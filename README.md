### DIRUncertainty
This repository contains code to predict the uncertainty of a given deformable vector field. It contains:
- Readers for various imaging formats (dicom,mha,nrrd,nifti,...) to read the input images (fixed and moving images) and corresponding vector fields.
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

  '''
  pip install -r requirements.txt
  '''
4. Download the model zoo with pretrained model weights
  '''
  TODO: make models available for download
  '''

## Usage
Run the runner.py script. In there, you can fill in the file paths to the fixed and moving images and also the DVF. Depending on your application, you can either save the results or run also the contour propagation or dose warping.
IMPORTANT NOTE: 
For contour propagation: the **moving** image is the image on which you have the contours, the **fixed** image is the one towards which you warp the contours.
For dose warping: the **moving** image is the image on which you have the dose, the **fixed** image is the one towards which you warp the dose.
