import numpy as np
from scipy import ndimage
import collections

import _hog


class HogError(Exception):
    pass


def hog(inImg, inPoints, nBins=9, blockSize=4, cellSize=4, interpolate=2):
    # to be reverse compatible
    _blockSize = blockSize
    _cellSize = cellSize

    res = []
    points = np.round(np.array(inPoints)).astype(np.int32)

    if len(inImg.shape) > 2:
        img = np.mean(inImg, axis=2).astype(np.uint8)
    else:
        img = np.array(inImg).astype(np.uint8)

    for point_id, point in enumerate(points):
        if isinstance(_blockSize, collections.Iterable):
            if isinstance(_blockSize[point_id], collections.Iterable):
                blockSize = _blockSize[point_id]
            else:
                blockSize = [_blockSize[point_id], _blockSize[point_id]]
            cellSize = _cellSize[point_id]
        else:
            blockSize = [_blockSize, _blockSize]
            cellSize = _cellSize

        gradsX, gradsY = _hog.gradientsAroundPoint(
            img, (point[0], point[1]), blockSize[0] * cellSize, blockSize[1] * cellSize
        )

        # print np.sum(np.abs(gradsY - gradsY2))

        angles = np.arctan2(gradsY, gradsX) * 180 / np.pi
        negInds = np.where(angles < 0)
        angles[negInds[0], negInds[1]] += 180
        magnitudes = np.sqrt(gradsY ** 2 + gradsX ** 2)

        # from matplotlib import pyplot as plt

        if interpolate == 2:
            cells = np.zeros((blockSize[1] + 2, blockSize[0] + 2, nBins))
        else:
            cells = np.zeros((blockSize[1], blockSize[0], nBins))
        for y in range(blockSize[1]):
            for x in range(blockSize[0]):
                test = getHistogram(magnitudes, angles, nBins, (y * cellSize, x * cellSize), cellSize, interpolate)
                if interpolate == 2:
                    cells[y : y + 3, x : x + 3] += test
                    # cells[i+1, j+1] = test[1, 1]
                else:
                    cells[x, y] = test

        if interpolate == 2:
            cells = cells[1:-1, 1:-1]
        cells = cells / np.linalg.norm(cells + 1e-9)
        res.append(cells.flatten())

    res = np.array(res)
    return res


def gradientsAroundPoint(img, point, blockSizeX, blockSizeY):
    xSobel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    ySobel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

    surroundingX = 1 + blockSizeX / 2
    surroundingY = 1 + blockSizeY / 2

    startY = point[1] - surroundingY
    endY = point[1] + surroundingY
    startX = point[0] - surroundingX
    endX = point[0] + surroundingX

    startXCorr = 0
    if startX < 0:
        startXCorr = -startX
        startX = 0
    startYCorr = 0
    if startY < 0:
        startYCorr = -startY
        startY = 0
    endXCorr = 2 * surroundingX
    if endX > img.shape[1] - 1:
        endXCorr = 2 * surroundingX - (endX - img.shape[1] + 1)
        endX = img.shape[1] - 1
    endYCorr = 2 * surroundingY
    if endY > img.shape[0] - 1:
        endYCorr = 2 * surroundingY - (endY - img.shape[0] + 1)
        endY = img.shape[0] - 1

    imgCut = np.zeros((2 * surroundingY, 2 * surroundingX))
    if startY < endY and startX < endX:
        imgCut[startYCorr:endYCorr, startXCorr:endXCorr] = img[startY:endY, startX:endX].astype(np.float32)

    gradsX = ndimage.filters.correlate(imgCut, xSobel)[1:-1, 1:-1] / 2
    gradsY = ndimage.filters.correlate(imgCut, ySobel)[1:-1, 1:-1] / 2

    # gradsX = ndimage.filters.sobel(imgCut)[1:-1, 1:-1]
    # gradsY = ndimage.filters.sobel(imgCut, axis=0)[1:-1, 1:-1]

    return gradsX, gradsY


def getHistogram(magnitudes, angles, nBins, start, cellSize, interpolate):
    # It was 9 instead of nBins in the function before refactoring.
    binValues = _hog.histogram(angles, magnitudes, nBins, (0, 180), interpolate, start, cellSize)

    if interpolate == 2:
        binValues = binValues.reshape((3, 3, nBins))
    return binValues


def getBinaryFeature(inImg, inPoints, features, forests, leafIndexMaps, nNodes):
    if features.shape[3] == 3:
        X = hog(inImg, inPoints, blockSize=features[0, 0, :, :2], cellSize=features[0, 0, :, 2])
    else:
        X = hog(inImg, inPoints, blockSize=features[:, 0, 0, 0], cellSize=features[:, 0, 0, 1])

    res = []
    for i in range(inPoints.shape[0]):
        temp = np.zeros(np.sum(nNodes[i]))
        for j in range(len(forests[i])):
            # for reverse compatibility with a tracker that only uses the local HOG feature for tree splits
            if np.any(forests[i][j].tree_.feature > 0):
                applied = forests[i][j].tree_.apply(X.reshape((1, 68, -1)))[0]
            else:
                applied = forests[i][j].tree_.apply(X[i].reshape((1, 1, -1)))[0]

            idx = leafIndexMaps[i][j][applied] + np.sum(nNodes[i][:j])
            temp[idx] = 1
        if i == 0:
            res = temp
        else:
            res = np.hstack((res, temp))

    # print res.shape
    return res


def getShiftFeature(inImg, inPoints, features):
    X = hog(inImg, inPoints, blockSize=features[0], cellSize=features[1])

    X = X.flatten()
    return X


def getRandomFeatures(radius, nFeatures):
    points = np.random.random_integers(-radius, radius, (nFeatures, 2))
    features = np.zeros((nFeatures, 4), dtype=np.int32)

    features[:, :2] = points
    features[:, 2] = 1
    features[:, 3] = 4

    return features
