import numpy as np


def getPixelArray(nifti):
    array = nifti.get_fdata()
    array = np.squeeze(array)  # get rid of time dimension
    if len(array.shape) == 4:
        array = np.transpose(array, [3, 0, 1, 2])
    return array


def getVoxelToWorldMatrix(nifti):
    voxelToWorldMatrix = nifti.get_affine()
    voxelToWorldMatrix[0, :] = -voxelToWorldMatrix[0, :]  # inverted convention?
    voxelToWorldMatrix[1, :] = -voxelToWorldMatrix[1, :]
    return voxelToWorldMatrix
