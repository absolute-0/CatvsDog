from Model import *
from ReadImage import *


def run_neural_net():
    print("getting data set")
    train_x, train_y, test_x, test_y = getDataSet()
    print("training")
    L_layer_model(train_x, train_y, print_cost=True)
    print("training done!!, predicting training data")
    predict(train_x, train_y)
    print("training done!!, predicting test data")
    p_test = predict(test_x, test_y)
    print("printing mislabeled images")
    print_mislabeled_images(classes, test_x, test_y, p_test)


if __name__ == '__main__':
    run_neural_net()
