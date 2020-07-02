import numpy as np
import time, os
import matplotlib.pyplot as plt
import matplotlib.pyplot as image
from PIL import Image
labels = {"cats": 0, "dogs": 1}
totalLabels = 2


def loadImage(filePath):
    # img = Image.open(filePath)
    # plt.imshow(img)
    # plt.show()
    # time.sleep(5)
    # plt.close()
    image1 = Image.open(filePath)
    if image1.size[0]>150 and image1.size[1]>150:
        return np.asarray(image1.resize((150,150)))
    # plt.imshow(image1)
    # plt.show()
    # time.sleep(4)
    # plt.close()
    return None


def showImage(data):
    plt.imshow(data)
    plt.show()


def loadImages(directory):
    images = []
    output = []
    global totalLabels
    minDimensions = np.array([9999, 9999, 3])
    totalabove100 = 0
    totalabove150 = 0
    totalabove200 = 0
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
                # minDimensions = np.minimum(minDimensions, currentImage.shape)
                # if currentImage.shape[0] >= 150 and currentImage.shape[1] >= 150:
                #     totalabove150 += 1
                # print(currentImage.shape)
    numpyImages = np.array(images)
    numpyOutput = np.array(output).reshape(1, len(output))
    # print(minDimensions,totalabove100,totalabove150,totalabove200)
    return numpyImages, numpyImages.reshape(len(images), -1).T, numpyOutput


def getDataSet():
    images, trainingSetX, trainingSetY = loadImages('Inputs/training_set/training_set')
    images, testSetX, testSetY = loadImages('Inputs/test_set/test_set')
    print (np.count_nonzero(trainingSetY==0))
    print(np.count_nonzero(testSetY == 0))
    print (trainingSetX.shape, trainingSetY.shape,testSetX.shape,testSetY.shape)
current_time = time.time()
getDataSet()
print (time.time()-current_time)

# loadImage('Inputs/')
