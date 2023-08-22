import pydicom
import numpy as np
import copy
import warnings
from skimage.draw import polygon2mask



ROI_COUNTOUR_SEQUENCE_INFO_TAG = "30060080"
ROI_CONTOUR_SEQUENCE_TAG = "30060040"
ROI_OBSERVATION_LABEL_TAG = "30060085"
ROI_NAME_TAG = "30060026"


def getPixelArray(dcm):
    array = np.transpose(dcm.pixel_array, axes=[2, 1, 0]).astype(float)
    if hasattr(dcm, "RescaleSlope"):
        array *= float(dcm.RescaleSlope)
    if hasattr(dcm, "RescaleIntercept"):
        array = array + float(dcm.RescaleIntercept)
    if hasattr(dcm, "DoseGridScaling"):
        array *= float(dcm.DoseGridScaling)
    return array


def getVoxelToWorldMatrix(dcm):
    voxelSpacing = getVoxelSpacing(dcm)
    imagePosition = np.array(dcm.ImagePositionPatient)
    imageOrientation = getImageOrientationPatient(dcm)
    assert np.all(imageOrientation == np.identity(3))
    orientationMatrix = imageOrientation * voxelSpacing
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, :3] = orientationMatrix
    voxelToWorldMatrix[:3, 3] = imagePosition
    return voxelToWorldMatrix


def slicedToSingleDicom(slicePaths):
    pixelArrayList = []
    sliceCoordinates = []
    baseDicom = pydicom.dcmread(slicePaths[0])
    for path in slicePaths:
        dcmSlice = pydicom.dcmread(path)
        if not hasattr(dcmSlice, "SliceThickness"):
            print("Pause")
        if isValidSlice(dcmSlice, baseDicom):
            pixelArrayList.append(
                convertStoredValuesToArray(dcmSlice.pixel_array, dcmSlice)
            )
            sliceCoordinates.append(dcmSlice.SliceLocation)
        else:
            raise ValueError("Invalid slice found at {}".format(path))

    assert isValidSliceList(sorted(sliceCoordinates), baseDicom.SliceThickness)
    pixelArrayList = [
        pixelArray for _, pixelArray in sorted(zip(sliceCoordinates, pixelArrayList))
    ]
    pixelArray = np.stack(pixelArrayList, axis=0)
    newPixelArrayPosition = getPixelArrayPosition(baseDicom, sliceCoordinates)
    return createDicom(pixelArray, baseDicom, newPixelArrayPosition)


def isValidSlice(dcmSlice, baseDcm):
    # TODO: add PID check, datetime check
    if not hasSameSliceThickness(dcmSlice, baseDcm):
        return False
    if not hasSamePixelSpacing(dcmSlice, baseDcm):
        return False
    if not hasSameSliceOrientation(dcmSlice, baseDcm):
        return False
    return True


def hasSameSliceThickness(dcmSlice, baseDcm):
    return dcmSlice.SliceThickness == baseDcm.SliceThickness


def hasSamePixelSpacing(dcmSlice, baseDcm):
    return dcmSlice.PixelSpacing == baseDcm.PixelSpacing


def getVoxelSpacing(dcm):
    voxelSpacing = list(dcm.PixelSpacing) + [dcm.SliceThickness]
    return np.array([float(spacing) for spacing in voxelSpacing])


def getImageOrientationPatient(dcm):
    # TODO: when going to non identity orientations, we have to check whether the vectors should be stacked horizontally or vertically
    orientation = dcm.ImageOrientationPatient
    xVec = np.array(orientation[:3])
    yVec = np.array(orientation[3:])
    zVec = np.cross(xVec, yVec)
    return np.stack([xVec, yVec, zVec])


def hasSameSliceOrientation(dcmSlice, baseDcm):
    return dcmSlice.ImageOrientationPatient == baseDcm.ImageOrientationPatient


def isValidSliceList(sortedSliceList, sliceThickness):
    sortedSliceList = np.array(sortedSliceList)
    diff = sortedSliceList[1:] - sortedSliceList[:-1]
    return np.allclose(diff, sliceThickness)


def getPixelArrayPosition(baseDicom, sliceCoordinates):
    pixelArrayPosition = baseDicom.ImagePositionPatient
    pixelArrayPosition[2] = min(sliceCoordinates)
    return pixelArrayPosition


def createDicom(newPixelArray, baseDicom, newPixelArrayPosition):
    baseDicom = copy.deepcopy(baseDicom)
    if hasattr(baseDicom, "InstanceNumber"):
        del baseDicom.InstanceNumber
    warnings.warn(
        "Dicoms are stored in zyx coordinates, please make sure the provided pixel array is as such"
    )
    pixelDataBytes = convertArrayToStoredValues(newPixelArray, baseDicom)
    baseDicom.NumberOfFrames, baseDicom.Rows, baseDicom.Columns = newPixelArray.shape
    baseDicom.PixelData = pixelDataBytes
    baseDicom.ImagePositionPatient = newPixelArrayPosition
    return baseDicom


def convertArrayToStoredValues(array, baseDicom):
    array = (array - baseDicom.RescaleIntercept) / (baseDicom.RescaleSlope)
    return convertToTypeWithWarning(array, baseDicom.pixel_array.dtype).tobytes()


def convertStoredValuesToArray(array, baseDicom):
    return array * (baseDicom.RescaleSlope) + baseDicom.RescaleIntercept


def convertToTypeWithWarning(array, targetType):
    convertedArray = convertToType(array, targetType)
    if not np.all(convertedArray == array):
        warnings.warn(
            "Conversion of array with type {} to {} changed some of its values".format(
                array.dtype, targetType
            )
        )
    return convertedArray


def convertToType(array, targetType):
    typeInfo = np.iinfo(targetType)
    array = np.clip(array, typeInfo.min, typeInfo.max)
    if isInteger(typeInfo):
        array = np.round(array)
    return array.astype(targetType)


def isInteger(typeInfo):
    return typeInfo.kind in "iub"

def convertContourSetToArray(dcm, voxelToWorldMatrix, shape):
    arrays = []
    for contour in getContourSequence(dcm):
        arrays.append(convertContourToArray(contour, voxelToWorldMatrix, shape))
    return np.stack(arrays, 0)


def convertContourSetToPointDict(dcm, voxelToWorldMatrix, shape):
    arrays = []
    for contour in getContourSequence(dcm):
        arrays.append(convertContourToPointDict(contour, voxelToWorldMatrix, shape))
    return arrays


def convertContourToArray(contour, voxelToWorldMatrix, shape):
    array = np.zeros(shape, dtype=np.int16)
    for contourSlice in contour.ContourSequence:
        points = np.array(contourSlice.ContourData).reshape(-1, 3)
        points = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
        newPoints = np.transpose(
            np.matmul(np.linalg.inv(voxelToWorldMatrix), np.transpose(points))
        )[:, :3]
        slice = round(newPoints[0, 2])
        if abs(slice - newPoints[0, 2]) > 10 ** -5:
            warnings.warn(
                f"Found a slice position at {newPoints[0,2]}, but set to {slice}"
            )

        array[:, :, slice] += polygon2mask(shape[:2], newPoints[:, :2])
    array = np.mod(array, 2)
    return array.astype(np.bool)


def convertContourToPointDict(contour, voxelToWorldMatrix, shape):
    pointDict = {}
    for contourSlice in contour.ContourSequence:
        points = np.array(contourSlice.ContourData).reshape(-1, 3)
        points = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
        newPoints = np.transpose(
            np.matmul(np.linalg.inv(voxelToWorldMatrix), np.transpose(points))
        )[:, :3]
        slice = int(newPoints[0, 2])
        if abs(slice - newPoints[0, 2]) > 10 ** -5:
            warnings.warn(
                f"Found a slice position at {newPoints[0,2]}, but set to {slice}"
            )
        if slice in pointDict:
            pointDict[slice].append(newPoints[:, :2])
        else:
            pointDict[slice] = [newPoints[:, :2]]
    return pointDict


def getContourSetNames(dcm):
    contourDataSequence = getContourData(dcm)
    contourNames = []
    for (_, contourInfo, structureInfo) in contourDataSequence:
        if ROI_NAME_TAG in structureInfo:
            contourNames.append(structureInfo.ROIName)
        else:
            contourNames.append(contourInfo.ROIObservationLabel)
    return contourNames


def getContourSetTypes(dcm):
    contourInfoSequence = getContourInfoSequence(dcm)
    return [contourInfo.RTROIInterpretedType for contourInfo in contourInfoSequence]


def changeContourSetNames(dcm, nameConversionDict):
    contourInfoSequence = getContourInfoSequence(dcm)
    for contourInfo in contourInfoSequence:
        oldName = contourInfo.ROIObservationLabel
        if oldName in nameConversionDict:
            contourInfo.ROIObservationLabel = nameConversionDict[oldName]
    return dcm


def getContourSequence(dcm):
    return [x for (x, _, _) in getContourData(dcm)]


def getContourInfoSequence(dcm):
    return [x for (_, x, _) in getContourData(dcm)]


def getStructureInfoSequence(dcm):
    return [x for (_, _, x) in getContourData(dcm)]


def getContourData(dcm):
    return [
        (contourObject, contourInfo, structureInfo)
        for (contourObject, contourInfo, structureInfo) in zip(
            dcm.ROIContourSequence,
            dcm[ROI_COUNTOUR_SEQUENCE_INFO_TAG],
            dcm.StructureSetROISequence,
        )
        if ROI_CONTOUR_SEQUENCE_TAG in contourObject
        and (ROI_OBSERVATION_LABEL_TAG in contourInfo or ROI_NAME_TAG in structureInfo)
    ]

