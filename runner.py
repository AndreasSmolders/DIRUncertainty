
from ContourPropagation.ContourWarper import ContourWarper
from ContourPropagation.structureSet import StructureSet
from DoseAccumulation.DoseMap import DoseMap
from DoseAccumulation.DoseWarper import DoseWarper
from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer
from Utils.scan import Scan
from Utils.vectorField import VectorField


# Required to run the uncertainty prediction
fixedImagePath = ''
movingImagePath = ''
vectorfieldPath = ''

# Optional depending on which task to perform
structureSetPath = None
dosePath = None

# The string contains the path to the folder with the pretrained model weights, 
# Patch size can be reduced if the GPU memory is too low to run inference. However,
# patches need to be stiched together so try not to go too low. The voxelSpacing should
# be kept at this value, it was used during training and affects the results
registerer = FixedMeanPatchRegisterer(
        'TrainedModels/L5S1-3', 
        patchSize=(256, 256, 96),
        voxelSpacing=(1.9531, 1.9531, 2.0),
    )

fixedImage = Scan(fixedImagePath)
movingImage = Scan(movingImagePath)
fixedDVF = VectorField(vectorfieldPath)


movedImage,probabilisticDVF = registerer.process(fixedImage,movingImage,fixedDVF)
# The output of the processsing is a probabilistic (Gaussian) vector field. It has a 
# 3xHxWxD mean field (same as the 'fixedDVF') and a 3xHxWxD standard deviation, which can be 
# accessed with 'probabilisticDVF.getVectorFieldUncertainty()'. Do not access the  
# probabilisticDVF.std attribute directly, as there is a weighting factor involved.

if structureSetPath is not None:
    structureSet = StructureSet(structureSetPath)
    structureprobabilities = ContourWarper(structureSet,probabilisticDVF).getProbabilisticWarpedStructures(50)
    # This contains an NxHxWxD array with N the number of structures. The voxel values depict the contour probabilities

if dosePath is not None:
    dose = DoseMap(dosePath)
    warpedDose = DoseWarper(dose,probabilisticDVF).getProbabilisticWarpedDose(50)
    # This contains an object of a 'SampledDoseMap', which is similar to a scan. It contains an array with an NxHxWxD 
    # with N the number of samples. Each sample is a sampled warped dose, which can be used in further analysis

