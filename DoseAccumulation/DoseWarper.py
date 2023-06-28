from warnings import warn
from DL.Model.gaussianBlur import get_gaussian_kernelNd
import numpy as np
from Utils.conversion import toNumpy
from Utils.array import concatenate, interpolate
from Utils.conversion import toTorch
from Utils.miscellaneous import getClassName
from Utils.vectorField import ProbabilisticVectorField, VectorField
from DoseAccumulation.DoseMap import DoseMap, SampledDoseMap
from Utils.warp import warpScan
import torch


class DoseWarper:
    """
    Class to warp a singe dosemap with a vectorfield
    inputs:
        doseMap : DoseMap
        vectorField : VectorField
    """

    def __init__(self, dosemap: DoseMap, vectorfield: VectorField):
        self.dosemap = dosemap
        self.vectorfield = vectorfield

    def getWarpedDose(self):
        return warpScan(self.getUnwarpedDoseMap(), self.getMostLikelyVectorField())

    def getUnwarpedDoseMap(self):
        return self.dosemap

    def getMostLikelyVectorField(self):
        return self.vectorfield.getVectorField()

    def getVectorFieldSample(self):
        if isinstance(self.vectorfield, ProbabilisticVectorField):
            return self.vectorfield.getVectorFieldSample()
        else:
            warn(
                f"Vectorfield of class {getClassName(self.vectorfield)} is not probabilistic, 'sampling' is deterministic"
            )
            return self.vectorfield

    def getWarpedDoseSample(self, vectorFieldSample=None):
        if vectorFieldSample is None:
            vectorFieldSample = self.getVectorFieldSample()
        return warpScan(self.getUnwarpedDoseMap(), vectorFieldSample)

    def getProbabilisticWarpedDose(self, nSamples=50, outputType=SampledDoseMap):
        doseMapSamples = []
        for _ in range(nSamples):
            doseMapSamples.append(toNumpy(self.getWarpedDoseSample().getPixelArray()))
        sampledDoseMap = SampledDoseMap(
            array=concatenate(doseMapSamples, 0),
            vtwMatrix=self.getUnwarpedDoseMap().getVoxelToWorldMatrix(),
        )
        if outputType != SampledDoseMap:
            return sampledDoseMap.getFittedDoseMap(outputType)
        else:
            return sampledDoseMap

