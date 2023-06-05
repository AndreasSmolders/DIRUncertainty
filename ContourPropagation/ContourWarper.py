from warnings import warn
from Utils.miscellaneous import getClassName
from ContourPropagation.structureSet import StructureSet
from Utils.vectorField import ProbabilisticVectorField, VectorField
from Utils.warp import warpArray, warpScan, warpStructureSet


class ContourWarper:
    """
    Class to warp a Structureset with a vectorfield
    inputs:
        strucuteSet : StructureSet
        vectorField : VectorField
    """

    def __init__(self, structureSet: StructureSet, vectorfield: VectorField):
        self.structureSet = structureSet
        self.vectorfield = vectorfield

    def getWarpedStructures(self):
        structureSet = warpStructureSet(
            self.getUnwarpedStructureSet(), self.getMostLikelyVectorField()
        )
        structureSet._setPixelArray(structureSet.getPixelArray() > 0.5)
        structureSet.updatePointDict()
        return structureSet

    def getUnwarpedStructureSet(self):
        return self.structureSet

    def getMostLikelyVectorField(self):
        return self.vectorfield.getVectorField()

    def getVectorFieldSample(self):
        if isinstance(self.vectorfield, ProbabilisticVectorField):
            return self.vectorfield.getVectorFieldSample()
        else:
            warn(
                f"Vectorfield of class {getClassName(self.vectorfield)} is not probabilistic, 'sampling' is deterministic"
            )
            return self.vectorfield.getVectorField()

    def getWarpedStructureSetSample(self, vectorFieldSample=None):
        if vectorFieldSample is None:
            vectorFieldSample = self.getVectorFieldSample()
        return warpStructureSet(self.getUnwarpedStructureSet(), vectorFieldSample)

    def getProbabilisticWarpedStructures(self, nSamples=50):
        structures = (self._getWarpedStructureSetArraySample() > 0.5).float() / nSamples
        for _ in range(1, nSamples):
            structures += (
                self._getWarpedStructureSetArraySample() > 0.5
            ).float() / nSamples
        return structures

    def _getWarpedStructureSetArraySample(self, vectorFieldSample=None):
        if vectorFieldSample is None:
            vectorFieldSample = self.getVectorFieldSample()
        return warpScan(
            self.getUnwarpedStructureSet(), vectorFieldSample
        ).getPixelArray()
