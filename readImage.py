import numpy as np
import time, os
import matplotlib.pyplot as plt
import matplotlib.pyplot as image
from PIL import Image
labels = {"cats": 0, "dogs": 1}
totalLabels = 2


def loadImage(filePath):
    image1 = Image.open(filePath)
    if image1.size[0]>150 and image1.size[1]>150:
        return np.asarray(image1.resize((150,150)))
    return None


def showImage(data):
    plt.imshow(data)
    plt.show()


def loadImages(directory):
    images = []
    output = []
    global totalLabels
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames[1:]:
            currentImage = loadImage(os.path.join(dirname, filename))
            if currentImage is not None:
                images.append(currentImage)
                labelName = dirname.split('/')[-1]
                if labelName not in labels:
                    labels[labelName] = totalLabels
                    totalLabels += 1
                output.append(labels[labelName])

    numpyImages = np.array(images)
    numpyOutput = np.array(output).reshape(1, len(output))
    return numpyImages, numpyImages.reshape(len(images), -1).T, numpyOutput


def getDataSet():
    images, trainingSetX, trainingSetY = loadImages('Inputs/training_set/training_set')
    images, testSetX, testSetY = loadImages('Inputs/test_set/test_set')
    print (np.count_nonzero(trainingSetY == 0))
    print(np.count_nonzero(testSetY == 0))
    print (trainingSetX.shape, trainingSetY.shape,testSetX.shape,testSetY.shape)
current_time = time.time()
getDataSet()
print (time.time()-current_time)
