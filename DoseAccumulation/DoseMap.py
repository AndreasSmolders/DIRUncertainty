from Utils.scan import Scan
from Utils.array import *
from Utils.vectorField import getNormalSample, getUniformSample
import os
import numpy as np

class DoseMap(Scan):
    def __init__(self, filePath=None, array=None, vtwMatrix=None):

        super().__init__(filePath=filePath, array=array, vtwMatrix=vtwMatrix)

    def getDoseMap(self):
        return self.getPixelArray()

    def getIntegralDose(self, mask):
        return (
            np.sum(self.getPixelArray()[mask])
            * np.product(self.getVoxelSpacing())
            / 10 ** 6
        )  # liter

    def getEUD(self, mask, exponent):
        return (np.mean((self.getPixelArray()[mask]) ** exponent)) ** (1 / exponent)

    def addDoseMap(self,doseMap):
        assert np.all(self.getVoxelToWorldMatrix() == doseMap.getVoxelToWorldMatrix())
        assert np.all(self.getShape() == doseMap.getShape())
        self._setPixelArray(self.getPixelArray()+doseMap.getPixelArray())


class ProbabilisticDoseMap(DoseMap):
    def getMeanDoseMap(self):
        return super().getDoseMap()

    def getDoseMapSample(self):
        raise NotImplementedError

    def getDoseMapUncertainty(self):
        raise NotImplementedError


class GaussianDoseMap(ProbabilisticDoseMap):
    def __init__(self, filePath=None, array=None, stdev=None, vtwMatrix=None):
        if filePath is not None:
            super().__init__(filePath=filePath)
            self.stdev = self.pixelArray[1:2]
            self.pixelArray = self.pixelArray[:1]
        elif stdev is None:
            self.pixelArray = array[:1]
            self.stdev = array[1:2]
            self.voxelToWorldMatrix = vtwMatrix
        else:
            self.pixelArray = array
            self.stdev = stdev
            self.voxelToWorldMatrix = vtwMatrix

    def getDoseMapUncertainty(self):
        return self.stdev

    def getDoseMapSample(self):
        mean = self.getMeanDoseMap()
        eps = getNormalSample(mean)
        return mean + self.getDoseMapUncertainty() * eps


class MinMaxDoseMap(ProbabilisticDoseMap):
    def getDoseMapUncertainty(self):
        return self.getMaxDoseMap() - self.getMinDoseMap()

    def getMinDoseMap(self):
        return self.getPixelArray()[1:2]

    def getMaxDoseMap(self):
        return self.getPixelArray()[2:3]

    def getMeanDoseMap(self):
        return self.getPixelArray()[0:1]

    def getDoseMapSample(self):
        mean = self.getMeanDoseMap()
        eps = getUniformSample(mean)
        return mean + self.getDoseMapUncertainty() * eps


class SampledDoseMap(ProbabilisticDoseMap):
    def getDoseMapUncertainty(self):
        return getStd(self.getDoseMap(), axis=0, keepdims=True, ddof=1)

    def getMeanDoseMap(self):
        return getMean(self.getDoseMap(), axis=0, keepdims=True)

    def getMinDoseMap(self):
        return getMin(self.getDoseMap(), axis=0, keepdims=True)

    def getMaxDoseMap(self):
        return getMax(self.getDoseMap(), axis=0, keepdims=True)

    def getDoseMapSample(self):
        n = np.random.randint(0, self.getDoseMap().shape[0])
        return self.getDoseMap()[n : n + 1]

    def getFittedDoseMap(self, fitType):
        if fitType == GaussianDoseMap:
            return GaussianDoseMap(
                array=self.getMeanDoseMap(),
                stdev=self.getDoseMapUncertainty(),
                vtwMatrix=self.voxelToWorldMatrix,
            )
        elif fitType == MinMaxDoseMap:
            return MinMaxDoseMap(
                array=concatenate(
                    (self.getMeanDoseMap(), self.getMinDoseMap(), self.getMaxDoseMap()),
                    0,
                ),
                vtwMatrix=self.voxelToWorldMatrix,
            )
        else:
            raise NotImplementedError

    def addSample(self, sample):
        if isinstance(sample, DoseMap):
            self._setPixelArray(
                concatenate((self.getPixelArray(), sample.getPixelArray()), 0)
            )
        else:
            self._setPixelArray(concatenate((self.getPixelArray(), sample), 0))
