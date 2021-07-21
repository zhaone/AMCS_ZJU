import numpy as np
import matplotlib.pyplot as plt

from svm import SVM

def dataGenerator(size, soft=False):
    data = np.random.rand(2, size) * 8 - 4

    label = data[0]**2 - data[1]**2 - 1
    tmpLabel = label.copy()
    label[tmpLabel < 0] = -1
    label[tmpLabel >= 0] = 1

    # add some pertubation
    if soft:
        mu, sigma = 0, 0.5
        data += np.random.normal(mu, sigma, (2, size))
    return data, label


def visTrain(data, label, svm):
    pSet = data[:, label == 1]
    nSet = data[:, label == -1]
    plt.scatter(pSet[0], pSet[1], c='b', marker='o', label='positive class')
    plt.scatter(nSet[0], nSet[1], c='y', marker='x', label='negtive class')

    pSupVec = svm.supData[:, svm.supLabel == 1]
    nSupVec = svm.supData[:, svm.supLabel == -1]
    plt.scatter(pSupVec[0], pSupVec[1], c='k', s=60,
                marker='o', label='positive support vector')
    plt.scatter(nSupVec[0], nSupVec[1], c='k', s=60,
                marker='x', label='negtive support vector')

    x1, x2 = np.meshgrid(np.linspace(-4, 4, 80), np.linspace(-4, 4, 80))
    X = np.array([[x1,  x2] for x1, x2 in zip(np.ravel(x1), np.ravel(x2))])
    pv, pl = svm.predict(X.T)
    pv = pv.reshape(80, 80)
    plt.contour(x1, x2, pv, [0.0], colors='r', linewidths=1,
                origin='lower', label='classificatoin boundary')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(567)
    trainSet, trainLabel = dataGenerator(100)
    testSet, fakeLabel = dataGenerator(100)
    svm = SVM()
    svm.fit(trainSet, trainLabel, kernel='gaussian')
    pValue, pLabel = svm.predict(testSet)
    visTrain(trainSet, trainLabel, svm)
