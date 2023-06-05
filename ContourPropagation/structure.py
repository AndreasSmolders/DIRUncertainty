from random import random
from typing import List
from Utils.array import concatenate, getDimensions, getMax
from Utils.conversion import toNumpy
from Utils.scan import Scan
from Utils.ioUtils import readJson
import numpy as np
from skimage.measure import find_contours

AUTOMATIC_ROI_ALGORITHM_LABEL = "AUTOMATIC"


class Structure(Scan):
    """
    General class to deal with structure sets
    """

    def __init__(
        self,
        filePath=None,
        array=None,
        vtwMatrix=None,
        structureName=None,
        structureType=None,
        pointDict=None,
        color=None,
    ):

        super().__init__(filePath=filePath, array=array, vtwMatrix=vtwMatrix)
        if not hasattr(self, "structureName"):
            self._setStructureName(structureName)
        if not hasattr(self, "structureType"):
            self._setStructureType(structureType)
        if not hasattr(self, "pointDict"):
            self.pointDict = pointDict
        if not hasattr(self, "color"):
            self.color = color

    def getStructureName(self):
        return self.structureName

    def getPointDict(self):
        if self.pointDict is None:
            self.updatePointDict()
        return self.pointDict

    def updatePointDict(self):
        self.pointDict = maskToPointdict(toNumpy(self.getPixelArray()))

    def getStructureType(self):
        return self.structureType

    def getColor(self):
        if not hasattr(self, "color"):
            self.color = None
        if self.color is None:
            return np.random.choice(COLORS)
        return self.color

    def getVolume(self):
        return (
            np.sum(self.getPixelArray()) * np.product(self.getVoxelSpacing()) / 10 ** 6
        )

    def _setStructureName(self, origName):
        self.structureName = origName

    def _setStructureType(self, structureType):
        self.structureType = structureType

    def _setPointDict(self, pointDict):
        self.pointDict = pointDict


def joinStructures(structures: List[Structure]):
    newArray = getMax(
        concatenate([structure.getPixelArray() for structure in structures], axis=0),
        axis=0,
        keepdims=True,
    )
    newPointDict = {}
    for structure in structures:
        if structure.getPointDict() is not None:
            for slice, pointList in structure.getPointDict().items():
                if slice in newPointDict:
                    newPointDict[slice] += pointList
                else:
                    newPointDict[slice] = pointList
    if len(newPointDict) == 0:
        newPointDict = None
    return Structure(
        array=newArray,
        structureName=structures[0].getStructureName(),
        structureType=structures[0].getStructureType(),
        vtwMatrix=structures[0].getVoxelToWorldMatrix(),
        pointDict=newPointDict,
        color=structures[0].getColor(),
    )


def maskToPointdict(mask):
    return _3DMaskToPointDict(mask[0])


def _3DMaskToPointDict(mask):
    pointDict = {}
    assert getDimensions(mask) == 3
    for i in range(mask.shape[-1]):
        if np.any(mask[:, :, i]):
            pointDict[i] = find_contours(mask[:, :, i], 0.5)
    return pointDict


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#808080"]


def hexToRGB(color):
    return [int(color[i : i + 2], 16) for i in (1, 3, 5)]