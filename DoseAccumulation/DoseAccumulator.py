import os
from typing import List, Union
from warnings import warn
from Model.gaussianBlur import get_gaussian_kernel, get_gaussian_kernelNd
from gate import readParticles
from ioUtils import isPickle, readPickle, writePickle
import numpy as np
from conversion import toNumpy
from plot3d import MagnitudeObject, plot3d
from Utils.array import concatenate, interpolate
from Utils.conversion import toArrayType, toTorch
from Utils.miscellaneous import getClassName
from Utils.timing import getTime, getTiming
from Utils.vectorField import ProbabilisticVectorField, VectorField
from DoseAccumulation.DoseMap import DoseMap, SampledDoseMap
from Utils.warp import warpDose, warpScan
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
        dosemaps: List[Union[str,DoseMap]],
        vectorfields: List[Union[str,VectorField]],
        mode="correlated",
    ):
        self.dosemaps = dosemaps
        self.vectorfields = vectorfields
        self.mode = mode
        assert self.mode in ["correlated", "uncorrelated"]
        '''
        The dosemaps can either contain DoseMap objects, or a path to where the the dose is saved.
        The vectorfields can either contain VectorField objects, or a path to where the the vf is saved.
        When giving paths, the accumulation requires less memory, but it will be slower. This can be useful when accumulating
        over a large number of fractions. The doses can be any file format. The vectorfields have to be a .pkl file
        if it is a probabilistic vector field, since there is no standard file format for these. If a non-probabilstic 
        vectorfield is used, other file formats are also permitted
        '''

    def getAccumulatedDose(self):
        accumulatedDose = None
        for dosePath, vectorFieldPath in zip(self.dosemaps, self.vectorfields):
            dose = self.toDose(dosePath)
            vectorField = self.toVectorField(vectorFieldPath)
            if accumulatedDose is None:
                accumulatedDose = warpDose(dose, vectorField)
            else:
                accumulatedDose.addDoseMap(warpDose(dose, vectorField))
        return accumulatedDose

    def getAccumulatedDoseSample(self):
        accumulatedDose = None

        if self.mode == "correlated":
            eps = self.toVectorField(self.vectorfields[0]).getStandardNormalSample()
        elif self.mode == "uncorrelated":
            eps = None

        for dosePath, vectorFieldPath in zip(self.dosemaps, self.vectorfields):
            dose = self.toDose(dosePath)
            vectorField = self.toVectorField(vectorFieldPath)

            vectorFieldSample = vectorField.getVectorFieldSample(eps)

            if accumulatedDose is None:
                accumulatedDose = warpDose(dose, vectorFieldSample)
            else:
                accumulatedDose.addDoseMap(warpDose(dose, vectorFieldSample))

        return accumulatedDose

    def getMultipleAccumulatedDoseSamples(self,saveFolder,nSamples = 50):
        '''
        The above function is generally usable and can be used if a one/a few samples are needed. 
        However, when generating more samples, this becomes inefficient, since loading a
        probabilistic vector field for each sample and fraction can be very slow.
        Instead, we can load the vf and dose of one fraction, and generate directly all the sampled 
        warped doses for that fraction. Then we go to the next fraction. The summed dose up to the 
        current fraction is loaded for each sample, current fraction is added, and the sum is 
        stored back on the disk. This is faster, but it requires to keep track of the EPS in case
        the VF samples of one accumulated dose need to be correlated
        '''
        sampleSeeds = np.random.randint(10000,99999,nSamples)

        for fractionNumber,(dosePath, vectorFieldPath) in enumerate(zip(self.dosemaps, self.vectorfields)):
            print(f'Creating samples for fraction {fractionNumber}')
            dose = self.toDose(dosePath)
            vectorField = self.toVectorField(vectorFieldPath)
            for sampleSeed in sampleSeeds:
                accumulatedDosePath = os.path.join(saveFolder,f'accumulatedDoseSample_{sampleSeed}.pkl')
                if self.mode == "correlated":
                    torch.manual_seed(sampleSeed)
                    eps = vectorField.getStandardNormalSample()
                    
                    writePickle(f'/data/user/smolde_a/HNCTest/DoseSamples/eps_{fractionNumber}_{sampleSeed}.pkl',eps)
                elif self.mode == "uncorrelated":
                    eps = None
                vectorFieldSample = vectorField.getVectorFieldSample(eps)

                warpedDoseSample = warpDose(dose, vectorFieldSample)
                if fractionNumber == 0:
                    warpedDoseSample.save(accumulatedDosePath)
                else:
                    accumulatedDose = DoseMap(accumulatedDosePath)
                    accumulatedDose.addDoseMap(warpedDoseSample)
                    accumulatedDose.save(accumulatedDosePath)

    def toDose(self,path):
        if isinstance(path,str):
            return DoseMap(path)
        return path
    
    def toVectorField(self,path):
        if isinstance(path,str):
            if isPickle(path):
                return readPickle(path)
            else:
                return VectorField(path)
        return path