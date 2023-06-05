from copy import deepcopy

import warnings
from Registration.patchRegistration import PatchRegisterer, makeScan, makeVectorField
from Utils.array import concatenate
from Utils.conversion import toTorch
from Utils.scan import Scan
from Utils.vectorField import GaussianVectorField, VectorField
from Utils.warp import warpScan


class FixedMeanPatchRegisterer(PatchRegisterer):
    def __init__(
        self,
        sessionDir,
        patchSize=None,
        voxelSpacing=None,
        device="cuda:0",
        dvfInput=True,
        applyPrewarping=False,
    ):
        super().__init__(sessionDir, patchSize, voxelSpacing, device)
        self.dvfInput = dvfInput
        self.applyPrewarping = applyPrewarping

    def process(self, fixedScan, movingScan, fixedVectorField):
        fixedScan = makeScan(fixedScan)
        movingScan = makeScan(movingScan)
        fixedVectorField = makeVectorField(fixedVectorField)
        inputImage = self.preProcess(fixedScan, movingScan, fixedVectorField)
        output = self.register(inputImage)
        movedScan, vectorField = self.postProcess(
            fixedScan, movingScan, fixedVectorField, output
        )
        return movedScan, vectorField

    def preProcess(self, fixedScan: Scan, movingScan: Scan, vectorField: VectorField):

        if self.applyPrewarping:
            movingScan = warpScan(movingScan, vectorField.getPixelArray())
        inputArray = super().preProcess(fixedScan, movingScan)
        if not self.dvfInput:
            return inputArray
        vectorField = self.resampleVectorField(
            vectorField, targetVoxelSpacing=self.voxelSpacing
        )
        vectorFieldArray = (
            toTorch(vectorField.getPixelArray()).float().unsqueeze(0).to(self.device)
        )

        return concatenate((inputArray, vectorFieldArray), 1)

    def postProcess(
        self, fixedScan: Scan, movingScan: Scan, fixedVectorField: VectorField, output
    ):
        vectorField = output
        vectorField = self.resampleVectorField(output, fixedScan.getShape())
        vectorField._setPixelArray(fixedVectorField.getPixelArray())
        vectorField.to(self.device)
        vectorField._setVoxelToWorldMatrix(fixedScan.getVoxelToWorldMatrix())
        movedScan = warpScan(deepcopy(movingScan), vectorField.getVectorField())
        return movedScan, vectorField


class MultiFixedMeanPatchRegisterer:
    def __init__(
        self,
        sessionDirs,
        patchSize=None,
        voxelSpacing=None,
        device="cuda:0",
        dvfInput=True,
        applyPrewarping=False,
    ):
        self.patchRegisterers = [
            FixedMeanPatchRegisterer(
                sessionDir, patchSize, voxelSpacing, device, dvfInput, applyPrewarping
            )
            for sessionDir in sessionDirs
        ]
        self.device = device

    def process(self, fixedScan, movingScan, fixedVectorField):

        movedScan, vectorField = self.patchRegisterers[0].process(
            fixedScan, movingScan, fixedVectorField
        )

        currentVectorFieldUncertainty = vectorField.getVectorFieldUncertainty()
        for alternativeRegisterer in self.patchRegisterers[1:]:
            _, alternativeVectorField = alternativeRegisterer.process(
                fixedScan, movingScan, fixedVectorField
            )
            assert type(vectorField) == type(alternativeVectorField)
            assert type(vectorField) == GaussianVectorField
            assert type(vectorField.samplingLayer) == type(
                alternativeVectorField.samplingLayer
            )
            if type(vectorField.samplingLayer.blurringLayer) != type(
                alternativeVectorField.samplingLayer.blurringLayer
            ) or vectorField.samplingLayer.blurringLayer.isEqual(
                alternativeVectorField.samplingLayer.blurringLayer
            ):
                warnings.warn(
                    "Vectorfields did not have the same blurring layer, the first one is retained."
                )
            alternativeVectorFieldUncertainty = (
                alternativeVectorField.getVectorFieldUncertainty()
            )
            currentVectorFieldUncertainty[
                alternativeVectorFieldUncertainty > currentVectorFieldUncertainty
            ] = alternativeVectorFieldUncertainty[
                alternativeVectorFieldUncertainty > currentVectorFieldUncertainty
            ]
        vectorField._setVectorFieldUncertainty(currentVectorFieldUncertainty)
        return movedScan, vectorField
