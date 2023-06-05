import numpy as np
from Utils.array import isArray
from Utils.constants import DEFAULT_DEVICE, PKL_FILE_EXTENSION
import Utils.mha as mhaLib
from Utils.miscellaneous import getClassAttributeKeys
import Utils.nrrdLib as nrrdLib
import Utils.nifti as niftiLib
import Utils.dicoms as dicom
from Utils.conversion import toNumpy, toTorch
from Utils.ioUtils import *
import torch
from Utils.transformations import getNormalToVoxelMatrix, getGrid


class Scan:
    """
    General class to deal with scans.
    Currently only supports axis aligned scans (no rotation  in voxel to world matrix)
    Arrays are stored as C,X,Y,Z, C = 1 for images, C = 3 for vectorfields
    """

    def __init__(self, filePath=None, array=None, vtwMatrix=None):
        if filePath is not None:
            # TODO: gz support
            if isDicom(filePath):
                self._loadDicom(filePath)
            elif isMha(filePath):
                self._loadMha(filePath)
            elif isMhd(filePath):
                self._loadMhd(filePath)
            elif isNumpy(filePath):
                self._loadNumpy(filePath)
            elif isPickle(filePath):
                self._loadPickle(filePath)
            elif isNrrd(filePath):
                self._loadNrrd(filePath)
            elif isNifti(filePath):
                self._loadNifti(filePath)
            elif isGz(filePath):
                with gz.GzipFile(filePath, "r") as f:
                    self.__init__(f)
            else:
                raise ValueError(f"Couldn't load filepath {filePath}")
        elif array is not None:
            self.pixelArray = array
            self.voxelToWorldMatrix = vtwMatrix
        self._setChannelMaybe()

    def getPixelArray(self):
        return self.pixelArray

    def getVoxelToWorldMatrix(self):
        return self.voxelToWorldMatrix

    def _setPixelArray(self, pixelArray):
        # Use with care
        self.pixelArray = pixelArray

    def _setVoxelToWorldMatrix(self, voxelToWorldMatrix):
        # Use with care
        self.voxelToWorldMatrix = voxelToWorldMatrix

    def _setOrigin(self, origin):
        self.voxelToWorldMatrix[:3, 3] = origin

    def _setVoxelSpacing(self, voxelSpacing):
        for i in range(3):
            self.voxelToWorldMatrix[i, i] = voxelSpacing[i]

    def getShape(self):
        return tuple(self.getPixelArray().shape[-3:])

    def getVoxelCoordinates(self, device=DEFAULT_DEVICE):
        return getGrid(getNormalToVoxelMatrix(self.getShape()), self.getShape(), device)

    def getWorldCoordinates(self, device=DEFAULT_DEVICE):
        return getGrid(self.getNormalToWorldMatrix(), self.getShape(), device)

    def getNormalToWorldMatrix(self):
        return np.matmul(
            self.getVoxelToWorldMatrix(), getNormalToVoxelMatrix(self.getShape())
        )

    def getVoxelSpacing(self):
        return getVoxelSpacing(self.voxelToWorldMatrix)

    def getOrigin(self):
        return getOrigin(self.voxelToWorldMatrix)

    def getOrientation(self):
        return getOrientation(self.voxelToWorldMatrix)

    def getFullResolutionArrayAttributes(self):  # needed e.g. for resampling
        attributes = []
        for attribute in getClassAttributeKeys(self):
            value = getattr(self, attribute)
            if isArray(value) and isSubTuple(self.getShape(), tuple(value.shape)):
                attributes.append(attribute)
        return attributes

    def to(self, device):
        for attribute in self.getFullResolutionArrayAttributes():
            setattr(self, attribute, toTorch(getattr(self, attribute)).to(device))

    def toNumpy(self):
        for attribute in self.getFullResolutionArrayAttributes():
            setattr(self, attribute, toNumpy(getattr(self, attribute)))

    def save(self, savePath):
        assert isPickle(
            savePath
        ), f"Scan objects must be saved with a '{PKL_FILE_EXTENSION}' extension"
        writePickle(savePath, self)

    def _loadDicom(self, dcmPath):
        dcm = readDicom(dcmPath)
        self.pixelArray = dicom.getPixelArray(dcm)
        self.voxelToWorldMatrix = dicom.getVoxelToWorldMatrix(dcm)

    def _loadMha(self, mhaPath):
        mha = readMha(mhaPath)
        self.pixelArray = mhaLib.getPixelArray(mha)
        self.voxelToWorldMatrix = mhaLib.getVoxelToWorldMatrix(mha)

    def _loadMhd(self, mhdPath):
        mhd = readMhd(mhdPath)
        self.pixelArray = mhaLib.getPixelArray(mhd)
        self.voxelToWorldMatrix = mhaLib.getVoxelToWorldMatrix(mhd)

    def _loadNrrd(self, nrrdPath):
        nrrd = readNrrd(nrrdPath)
        self.pixelArray = nrrdLib.getPixelArray(nrrd)
        self.voxelToWorldMatrix = nrrdLib.getVoxelToWorldMatrix(nrrd)

    def _loadNifti(self, niftiPath):
        nifti = readNifti(niftiPath)
        self.pixelArray = niftiLib.getPixelArray(nifti)
        self.voxelToWorldMatrix = niftiLib.getVoxelToWorldMatrix(nifti)

    def _loadNumpy(self, numpyPath):
        self.pixelArray = np.load(numpyPath)
        self.voxelToWorldMatrix = np.identity(4)

    def _loadPickle(self, picklePath):
        scanObj = readPickle(picklePath)
        self.__dict__.update(scanObj.__dict__)

    def _setChannelMaybe(self):
        if len(self.pixelArray.shape) == 3:
            if isinstance(self.pixelArray, torch.Tensor):
                self.pixelArray = self.pixelArray.unsqueeze(0)
            else:
                self.pixelArray = np.expand_dims(self.pixelArray, 0)


def readPixelArray(path):
    return Scan(path).getPixelArray()


def getVoxelSpacing(voxelToWorldMatrix):
    return np.diagonal(voxelToWorldMatrix)[:3]


def getOrientation(voxelToWorldMatrix):
    return voxelToWorldMatrix[:3, :3] / np.diagonal(voxelToWorldMatrix)[:3]


def getOrigin(voxelToWorldMatrix):
    return voxelToWorldMatrix[:3, 3]


def isSubTuple(tup1, tup2):
    return " ".join(map(str, tup1)) in " ".join(map(str, tup2))


def toArray(scan):
    if isinstance(scan, Scan):
        return scan.getPixelArray()
    else:
        return scan
