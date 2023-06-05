from typing import List
import numpy as np
from Utils.conversion import toNumpy
from Utils.array import concatenate, interpolate
from Utils.conversion import toArrayType, toTorch
from Utils.miscellaneous import getClassName
from Utils.vectorField import ProbabilisticVectorField, VectorField
from DoseAccumulation.DoseMap import DoseMap, SampledDoseMap
from Utils.warp import warpScan,warpDose
import torch


class DoseAccumulator:
    """
    Class to accumulate a dosemaps with  vectorfields
    inputs:
        doseMap : DoseMap
        vectorField : VectorField
    """

    def __init__(
        self,
        dosemaps: List[DoseMap],
        vectorfields: List[VectorField],
        mode="correlated",
    ):
        self.dosemaps = dosemaps
        self.vectorfields = vectorfields
        self.mode = mode
        assert self.mode in ["correlated", "uncorrelated"]

    def getAccumulatedDose(self):
        accumulatedDose = toArrayType(
            np.zeros([1, *self.vectorfields[0].getShape()]),
            self.vectorfields[0].getPixelArray(),
        )
        for dose, vectorField in zip(self.dosemaps, self.vectorfields):
            accumulatedDose += toArrayType(
                warpDose(dose, vectorField).getPixelArray(), accumulatedDose
            )
        return DoseMap(
            array=accumulatedDose,
            vtwMatrix=self.vectorfields[0].getVoxelToWorldMatrix(),
        )

    def getAccumulatedDoseSample(self):
        accumulatedDose = toArrayType(
            np.zeros([1, *self.vectorfields[0].getShape()]),
            self.vectorfields[0].getPixelArray(),
        )

        if self.mode == "correlated":
            eps = self.vectorfields[0].getStandardNormalSample()
        elif self.mode == "uncorrelated":
            eps = None

        for dose, vectorField in zip(self.dosemaps, self.vectorfields):
            vectorFieldSample = vectorField.getVectorFieldSample(eps)
            accumulatedDose += warpScan(dose, vectorFieldSample).getPixelArray()

        return DoseMap(
            array=accumulatedDose,
            vtwMatrix=self.vectorfields[0].getVoxelToWorldMatrix(),
        )

    def getProbabilisticAccumulatedDose(self, nSamples=50, outputType=SampledDoseMap):
        doseMapSamples = []
        for _ in range(nSamples):
            doseMapSamples.append(
                toNumpy(self.getAccumulatedDoseSample().getPixelArray())
            )
            print("done")
        sampledDoseMap = SampledDoseMap(
            array=concatenate(doseMapSamples, 0),
            vtwMatrix=self.vectorfields[0].getVoxelToWorldMatrix(),
        )
        if outputType != SampledDoseMap:
            return sampledDoseMap.getFittedDoseMap(outputType)
        else:
            return sampledDoseMap
