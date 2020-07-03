import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

labels = {"cats": 0, "dogs": 1}
classes = {0 : "cats", 1 : "dogs"}
totalLabels = 2


def load_image(file_path):
    image1 = Image.open(file_path)
    if image1.size[0] > 150 and image1.size[1] > 150:
        return np.asarray(image1.resize((100, 100)))
    return None


def show_image(data):
    plt.imshow(data)
    plt.show()


def load_images(directory):
    images = []
    output = []
    global totalLabels
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames[1:]:
            current_image = load_image(os.path.join(dirname, filename))
            if current_image is not None:
                images.append(current_image)
                label_name = dirname.split('/')[-1]
                if label_name not in labels:
                    labels[label_name] = totalLabels
                    classes[totalLabels] = label_name
                    totalLabels += 1
                output.append(labels[label_name])

    numpy_images = np.array(images)
    numpy_output = np.array(output).reshape(1, len(output))
    return numpy_images, numpy_images.reshape(len(images), -1).T, numpy_output


def getDataSet():
    images, training_set_x, training_set_y = load_images('Inputs/training_set/training_set')
    images, test_set_x, test_set_y = load_images('Inputs/test_set/test_set')
    print(np.count_nonzero(training_set_y == 0))
    print(np.count_nonzero(test_set_y == 0))
    print("training_set_x  shape : ", training_set_x.shape, "training_set_y  shape : ", training_set_y.shape,
          "test_set_x  shape : ", test_set_x.shape, "test_set_y  shape : ", test_set_y.shape)
    return (training_set_x/255, training_set_y, test_set_x/255, test_set_y)


if __name__ == '__main__':
    current_time = time.time()
    getDataSet()
    print(time.time() - current_time)
