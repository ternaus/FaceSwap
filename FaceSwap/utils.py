from FaceSwap import NonLinearLeastSquares
import cv2
from FaceSwap import models
import numpy as np


def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ


def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]


def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)

    return mesh


def getShape3D(mean3DShape, blendshapes, params):
    # skalowanie
    s = params[0]
    # rotacja
    r = params[1:4]
    # przesuniecie (translacja)
    t = params[4:6]
    w = params[6:]

    # macierz rotacji z wektora rotacji, wzor Rodriguesa
    R = cv2.Rodrigues(r)[0]
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D


def getMask(renderedImg):
    return np.zeros(renderedImg.shape[:2], dtype=np.uint8)


def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D


def getFaceKeypoints(img, detector, predictor, inputLandmarks=None):
    if inputLandmarks is None:
        # detekcja twarzy
        dets = detector(img, 1)

        if len(dets) == 0:
            return None
        det = dets[0]
        inputLandmarks = bestFitRect(None, predictor.initLandmarks, [det.left(), det.top(), det.right(), det.bottom()])

    shapes2D = []
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)

    shape2D = predictor.processImg(img[np.newaxis], inputLandmarks)

    # transpozycja, zeby ksztalt byl 2 x n a nie n x 2, pozniej ulatwia to obliczenia
    shape2D = shape2D.T

    shapes2D.append(shape2D)

    return shapes2D


def getFaceTextureCoords(img, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor):
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    keypoints = getFaceKeypoints(img, detector, predictor)[0]
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(
        modelParams,
        projectionModel.residual,
        projectionModel.jacobian,
        ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]),
        verbose=0,
    )
    return projectionModel.fun([mean3DShape, blendshapes], modelParams)


def bestFit(destination, source, returnTransform=False):
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i]
    b = b / np.linalg.norm(srcVec) ** 2

    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean

    return np.dot(srcVec.reshape((-1, 2)), T) + destMean


def bestFitRect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]
    S0 += boxCenter - S0Center

    return S0


def getState(blendshapeWeights, pose):
    eyesClosedIdx = 5
    eyesClosedThreshold = 0.85
    smileIdx = 2
    smileThreshold = 0.45
    browIdx = 3
    browThreshold = -0.35 - np.sin(pose[0] / 3)
    mouthOpenIndex = 1
    mouthOpenThreshold = 0.4

    # print([blendshapeWeights[browIdx], browThreshold])

    eyesClosed = False
    if blendshapeWeights[eyesClosedIdx] > eyesClosedThreshold:
        eyesClosed = True

    smile = False
    if blendshapeWeights[smileIdx] > smileThreshold:
        smile = True

    browRaised = False
    if blendshapeWeights[browIdx] < browThreshold:
        browRaised = True

    mouthOpen = False
    if blendshapeWeights[mouthOpenIndex] > mouthOpenThreshold:
        mouthOpen = True

    return [eyesClosed, smile, browRaised, mouthOpen]
