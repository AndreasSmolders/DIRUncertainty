from copy import deepcopy
import torch
import torch.nn.functional as nnf
from Utils.constants import DEFAULT_DEVICE
from Utils.conversion import toArrayType, toNumpy, toTorch
from Utils.array import getDimensions
from Utils.interpolate import resampleScan
from Utils.scan import Scan
from Utils.transformations import transformVectorarray
import numpy as np
from Utils.vectorField import VectorField

METHOD_TORCH = "torch"

def warpStructureSet(structureSet, vectorArray):
    structureSet = warpScan(
        structureSet,
        vectorArray,
    )
    structureSet.updatePointDict()
    return structureSet


def warpDose(dose, vectorArray, method=METHOD_TORCH, device=DEFAULT_DEVICE):
    return warpScan(dose, vectorArray, method, device)


def warpScan(scan: Scan, vectorArray, method=METHOD_TORCH, device=DEFAULT_DEVICE):
    if isinstance(vectorArray, VectorField):
        scan = deepcopy(scan)
        return warpScanWithVectorField(scan, vectorArray, device)
    else:  # should phase out.
        scan = deepcopy(scan)
        warpedArray = warpArray(scan.getPixelArray(), vectorArray, method)
        scan._setPixelArray(warpedArray)
        return scan


def warpScanWithTransformationMatrix(
    scan: Scan, transformationMatrix, targetVtwMatrix=None, targetShape=None
):
    warpedScan = deepcopy(scan)
    if targetVtwMatrix is None:
        targetVtwMatrix = warpedScan.getVoxelToWorldMatrix()
    if targetShape is None:
        targetShape = warpedScan.getShape()
    warpedScan._setVoxelToWorldMatrix(
        np.matmul(transformationMatrix, warpedScan.getVoxelToWorldMatrix())
    )
    return resampleScan(warpedScan, targetVtwMatrix, targetShape)


def warpArray(imageArray, vectorArray, method=METHOD_TORCH):
    if method == METHOD_TORCH:
        vectorArray = toTorch(vectorArray)
        warpedArray = warpTorch(
            toArrayType(imageArray, vectorArray).float(), vectorArray
        )
        if not isinstance(imageArray, torch.Tensor):
            return toNumpy(warpedArray)
        return warpedArray
    else:
        raise NotImplementedError


def warpTorch(imageArray, vectorArray, mode="bilinear"):
    dim = getDimensions(imageArray)
    shape = imageArray.shape[-3:]
    grid = getMeshGridTorch(shape).to(vectorArray.device)
    newLocations = grid + vectorArray.float()

    # need to normalize grid values to [-1, 1] for resampler
    for i in range(len(shape)):
        newLocations[i] = 2 * (newLocations[i] / (shape[i] - 1) - 0.5)

    newLocations = newLocations.permute(1, 2, 3, 0)
    newLocations = newLocations[..., [2, 1, 0]]

    newLocations = newLocations.unsqueeze(0)
    imageArray = imageArray.unsqueeze(0)
    if dim == 3:
        return nnf.grid_sample(
            imageArray.unsqueeze(0), newLocations, align_corners=True, mode=mode
        )[0, 0]
    else:
        return nnf.grid_sample(imageArray, newLocations, align_corners=True, mode=mode)[
            0
        ]


def getMeshGridTorch(shape):
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    return grid.float()


def warpScanWithVectorField(
    scan: Scan, vectorField: VectorField, device=DEFAULT_DEVICE
):
    scan.to(device)
    if not vectorField.inMM:
        vectorField.toMM()
    worldTransformation = (
        vectorField.getWorldCoordinates(device)
        + toTorch(vectorField.getPixelArray()).to(device)
    ).float()
    worldToNormalMatrix = (
        toTorch(np.linalg.inv(scan.getNormalToWorldMatrix())).to(device).float()
    )
    normalTransformation = transformVectorarray(
        worldToNormalMatrix, worldTransformation
    )
    normalTransformation = torch.permute(
        normalTransformation[[2, 1, 0]], (1, 2, 3, 0)
    ).unsqueeze(0)
    newArray = nnf.grid_sample(
        scan.getPixelArray().float().unsqueeze(0),
        normalTransformation,
        align_corners=True,
        mode="bilinear",
    )[0]
    scan._setVoxelToWorldMatrix(vectorField.getVoxelToWorldMatrix())
    scan._setPixelArray(newArray)
    return scan
