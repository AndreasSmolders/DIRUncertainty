from Utils.array import concatenate, getMax
import copy
from Utils.dicoms import (
    ROI_COUNTOUR_SEQUENCE_INFO_TAG,
    convertContourSetToArray,
    convertContourSetToPointDict,
    getContourSetNames,
    getContourSetTypes,
)
from Utils.scan import Scan, readPixelArray
from Utils.ioUtils import getFileName, isPickle, readDicom, readPickle
import numpy as np
from ContourPropagation.structure import Structure, hexToRGB, joinStructures

AUTOMATIC_ROI_ALGORITHM_LABEL = "AUTOMATIC"
OLD_STRUCTURENAMES_KEY = "structureNames"
OLD_STRUCTURETYPES_KEY = "structureTypes"
OLD_PIXELARRAY_KEY = "pixelArray"
OLD_POINTDICT_KEY = "pointDict"


class StructureSet(Scan):
    """
    General class to deal with structure sets
    """

    def __init__(
        self,
        filePath=None,
        referenceScan=None,
        structures=None,
        vtwMatrix=None,
    ):
        self.structures = []
        self.voxelToWorldMatrix = None
        if filePath is not None:
            if isinstance(filePath, list):
                self._loadFromFileList(filePath, referenceScan)
            elif isPickle(filePath):
                self._loadPickle(filePath)
                self.structures = [
                    Structure(
                        array=structure.getPixelArray(),
                        vtwMatrix=structure.getVoxelToWorldMatrix(),
                        structureName=structure.getStructureName(),
                        structureType=structure.getStructureType(),
                        pointDict=structure.getPointDict(),
                        color=structure.getColor(),
                    )
                    for structure in self.structures
                ]
            else:
                self._loadDicomContourSet(filePath, referenceScan)
        else:
            if structures is not None:
                self.structures = structures
            if vtwMatrix is not None:
                self._setVoxelToWorldMatrix(vtwMatrix)
        self.sort()

    def getStructure(self, structureName=None, combineStructures=True):
        if structureName is None:
            return self.structures
        elif isinstance(structureName, list):
            return [
                self._getSingleStructure(sn, combineStructures) for sn in structureName
            ]
        else:
            return self._getSingleStructure(structureName, combineStructures)

    def getPixelArray(self, structureName=None, combineStructures=True):
        structure = self.getStructure(structureName, combineStructures)
        if isinstance(structure, list):
            return concatenate([s.getPixelArray() for s in structure], axis=0)
        else:
            return structure.getPixelArray()

    def getPointDict(self, structureName=None, combineStructures=True):
        structure = self.getStructure(structureName, combineStructures)
        if isinstance(structure, list):
            return [s.getPointDict() for s in structure]
        else:
            return structure.getPointDict()

    def getStructureNames(self):
        return [structure.getStructureName() for structure in self.getStructure()]

    def getStructureTypes(self):
        return [structure.getStructureType() for structure in self.getStructure()]

    def updatePointDict(self):
        for structure in self.getStructure():
            structure.updatePointDict()

    def __len__(self):
        return len(self.getStructure())

    def sort(self):
        order = np.argsort(self.getStructureNames())
        self.structures = [self.structures[i] for i in order]

    def addStructure(
        self,
        structure=None,
        pixelArray=None,
        structureName=None,
        structureType=None,
        pointDict=None,
    ):
        if structure is not None:
            assert isinstance(structure, Structure)
            self.structures.append(structure)
        else:
            if len(self.getStructure()) > 0:
                assert pixelArray.shape[-3:] == self.getShape()
            self.structures.append(
                Structure(
                    array=pixelArray,
                    vtwMatrix=self.getVoxelToWorldMatrix(),
                    structureName=structureName,
                    structureType=structureType,
                    pointDict=pointDict,
                )
            )
        self.sort()

    def removeStructure(self, structureName):
        self.structures = [
            structure
            for structure in self.structures
            if structure.structureName != structureName
        ]

    def filterStructures(self, structureNames, strict=True):
        indices = [
            i
            for i, structureName in enumerate(self.getStructureNames())
            if structureName in structureNames
        ]
        if strict:
            difference = set(structureNames) - set(self.getStructureNames())
            if len(difference) > 0:
                raise ValueError(
                    f"Not all structures could be found in the structureset: {difference}"
                )
        self.structures = [self.structures[i] for i in indices]

    def combineRepeatedStructures(self):
        uniqueStructures = set(self.getStructureNames())
        for structureName in uniqueStructures:
            structures = [
                self.structures[i]
                for i, x in enumerate(self.structures)
                if x.getStructureName() == structureName
            ]
            if len(structures) > 1:
                self.removeStructure(structureName)
                self.addStructure(joinStructures(structures))

    def toNumpy(self):
        return [structure.toNumpy() for structure in self.getStructure()]

    def to(self, device):
        return [structure.to(device) for structure in self.getStructure()]

    def _getSingleStructure(self, structureName, combineStructures=True):
        if structureName not in self.getStructureNames():
            raise ValueError(
                f"Could not find a structure in this set with name {structureName}"
            )
        if combineStructures:
            structures = [
                self.structures[i]
                for i, x in enumerate(self.structures)
                if x.getStructureName() == structureName
            ]
            return joinStructures(structures)

        else:
            index = self._getStructureNameIndex(structureName)
            return self.structures[index]

    def _getStructureNameIndex(self, structureName):
        indices = [
            i for i, x in enumerate(self.getStructureNames()) if x == structureName
        ]
        if len(indices) > 1:
            raise ValueError(
                f"Structure {structureName} is occuring multiple times in the structureSet"
            )
        elif len(indices) == 0:
            raise ValueError(
                f"Could not find a structure in this set with name {structureName}"
            )
        else:
            return indices[0]

    def _loadDicomContourSet(self, filePath, referenceScan: Scan):
        vtwMatrix = referenceScan.getVoxelToWorldMatrix()
        shape = referenceScan.getShape()
        dcm = readDicom(filePath)
        pixelArray = convertContourSetToArray(dcm, vtwMatrix, shape)

        self._setVoxelToWorldMatrix(vtwMatrix)
        structureNames = getContourSetNames(dcm)
        structureTypes = getContourSetTypes(dcm)
        pointDicts = convertContourSetToPointDict(dcm, vtwMatrix, shape)
        self.structures = []
        for i in range(len(structureNames)):
            self.addStructure(
                pixelArray=pixelArray[i : i + 1],
                structureName=structureNames[i],
                structureType=structureTypes[i],
                pointDict=pointDicts[i],
            )

    def _loadPickle(self, picklePath):
        scanObj = readPickle(picklePath)
        if self.__dict__.keys() == scanObj.__dict__.keys():
            self.__dict__.update(scanObj.__dict__)
        else:
            keys = scanObj.__dict__.keys()
            if (
                OLD_STRUCTURENAMES_KEY in keys
                and OLD_STRUCTURETYPES_KEY in keys
                and OLD_PIXELARRAY_KEY in keys
            ):
                print("Loading old StructureSet format")
                self._setVoxelToWorldMatrix(scanObj.getVoxelToWorldMatrix())
                for i in range(len(scanObj.__dict__[OLD_STRUCTURENAMES_KEY])):
                    if (
                        OLD_POINTDICT_KEY in keys
                        and scanObj.__dict__[OLD_POINTDICT_KEY] is not None
                    ):
                        pointDict = scanObj.__dict__[OLD_POINTDICT_KEY][i]
                    else:
                        pointDict = None
                    self.structures.append(
                        Structure(
                            array=scanObj.__dict__[OLD_PIXELARRAY_KEY][i : i + 1],
                            structureName=scanObj.__dict__[OLD_STRUCTURENAMES_KEY][i],
                            structureType=scanObj.__dict__[OLD_STRUCTURETYPES_KEY][i],
                            vtwMatrix=scanObj.getVoxelToWorldMatrix(),
                            pointDict=pointDict,
                        )
                    )
                print("Loading old StructureSet successful, overwriting")
                self.save(picklePath)

    def _loadFromFileList(self, fileList, referenceScan: Scan):
        self._setVoxelToWorldMatrix(referenceScan.getVoxelToWorldMatrix())
        for k in fileList:
            array = readPixelArray(k).astype(np.bool)
            structureName = getFileName(k)
            if "CTV" in structureName:
                structureType = "CTV"
            elif "PTV" in structureName:
                structureType = "PTV"
            elif "GTV" in structureName:
                structureType = "GTV"
            else:
                structureType = "ORGAN"
            self.structures.append(
                Structure(
                    array=array,
                    structureName=structureName,
                    structureType=structureType,
                )
            )

    def _setPixelArray(self, pixelArray):
        assert pixelArray.shape[0] == self.__len__()
        for i, structure in enumerate(self.structures):
            structure._setPixelArray(pixelArray[i : i + 1])

    def _setStructureNames(self, structureNames):
        assert len(structureNames) == self.__len__()
        for i, structure in enumerate(self.structures):
            structure._setStructureName(structureNames[i])

    def _setStructureTypes(self, structureTypes):
        assert len(structureTypes) == self.__len__()
        for i, structure in enumerate(self.structures):
            structure._setStructureType(structureTypes[i])

    def _setVoxelToWorldMatrix(self, voxelToWorldMatrix):
        super()._setVoxelToWorldMatrix(voxelToWorldMatrix)
        if hasattr(self, "structures"):
            for structure in self.structures:
                structure._setVoxelToWorldMatrix(voxelToWorldMatrix)


def writeRTStructDicom(structureSet: StructureSet, referenceDicom, targetPath):
    dcm = copy.deepcopy(referenceDicom)
    baseContour = referenceDicom.ROIContourSequence[0].ContourSequence[0]
    baseROIContour = referenceDicom.ROIContourSequence[0]
    baseROIContourInfo = referenceDicom[ROI_COUNTOUR_SEQUENCE_INFO_TAG][0]
    baseROIStructureSetInfo = referenceDicom.StructureSetROISequence[0]
    newROIContourSequence = []
    newROIContourInfoSequence = []
    newROIStructureSetInfoSequence = []
    for i, structure in enumerate(structureSet.getStructure()):
        assert isinstance(structure, Structure)
        newROIContour = copy.deepcopy(baseROIContour)
        newROIContourInfo = copy.deepcopy(baseROIContourInfo)
        newROIStructureSetInfo = copy.deepcopy(baseROIStructureSetInfo)

        contourSequence = []
        for slice, voxelcoordinateList in structure.getPointDict().items():
            for voxelCoordinates in voxelcoordinateList:
                newContour = copy.deepcopy(baseContour)
                nPoints = voxelCoordinates.shape[0]
                worldCoordinates = voxelCoordinatesToWorld(
                    np.concatenate(
                        (voxelCoordinates, slice * np.ones((nPoints, 1))), 1
                    ),
                    structureSet.getVoxelToWorldMatrix(),
                )
                newContour.ContourData = list(worldCoordinates.reshape(-1))
                newContour.NumberOfContourPoints = str(nPoints)
                contourSequence.append(newContour)
        newROIContour.ContourSequence = contourSequence
        newROIContour.ReferencedROINumber = str(i)
        newROIContour.ROIDisplayColor = hexToRGB(structure.getColor())
        newROIContourSequence.append(newROIContour)

        newROIContourInfo.ObservationNumber = str(i)
        newROIContourInfo.ReferencedROINumber = str(i)
        newROIContourInfo.ROIObservationLabel = structure.getStructureName()
        if structure.getStructureType() is not None:
            newROIContourInfo.RTROIInterpretedType = structure.getStructureType()
        newROIContourInfoSequence.append(newROIContourInfo)

        newROIStructureSetInfo.ROINumber = str(i)
        newROIStructureSetInfo.ROIName = structure.getStructureName()
        newROIStructureSetInfo.ROIDescription = structure.getStructureName()
        newROIStructureSetInfo.ROIGenerationAlgorithm = AUTOMATIC_ROI_ALGORITHM_LABEL
        newROIStructureSetInfoSequence.append(newROIStructureSetInfo)

    dcm.ROIContourSequence = newROIContourSequence
    dcm.RTROIObservationsSequence = newROIContourInfoSequence
    dcm.StructureSetROISequence = newROIStructureSetInfoSequence
    dcm.save_as(targetPath)

def voxelCoordinatesToWorld(coordinates, voxelToWorldMatrix):
    if len(coordinates.shape) == 1:
        coordinates = np.expand_dims(coordinates, 0)
        reshape = True
    else:
        reshape = False
    coordinates = np.concatenate((coordinates, np.ones((coordinates.shape[0], 1))), 1)
    coordinates = np.transpose(
        np.matmul(voxelToWorldMatrix, np.transpose(coordinates))
    )[:, :3]
    if reshape:
        return coordinates[0]
    else:
        return coordinates