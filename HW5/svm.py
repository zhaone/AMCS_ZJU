import numpy as np

from util import asm, kernelFunc

class SVM():
    def __init__(self):

        self.kernel = 'linear'
        self.supNum = 0
        self.supAlpha = None
        self.supData = None
        self.supLabel = None
        self.b = None

    def fit(self, data, label, kernel='linear'):
        self.kernel = kernel
        H, c, aI, bI, aE, bE = self.prepare(data, label)
        initX, initASIdx = self.trickInit(aI, bI, label)
        alpha = asm(initX, initASIdx, H, c, aI, bI, aE, bE, endThr=10e-5, maxIter=200)
        supMask = np.squeeze(alpha > 1e-5)
        self.supNum = np.sum(supMask)
        self.supAlpha = np.squeeze(alpha[supMask])
        self.supData = data[:, supMask]
        self.supLabel = label[supMask]
        self.b = np.mean(self.supLabel - \
                         np.sum(kernelFunc(self.supData, self.supData, kernel)*self.supLabel*self.supAlpha, axis=0))

    def predict(self, testData):
        # testData: (dim, tn)
        # self.supData: (dim, sn)
        label = np.zeros(testData.shape[1])
        kernelValue = kernelFunc(testData, self.supData, self.kernel) #(tn, sn)
        clsValue = kernelValue*self.supAlpha*self.supLabel + self.b
        clsValue = np.sum(clsValue, axis=1)
        label[clsValue > 0] = 1
        label[clsValue < 0] = -1
        return clsValue, label

    def prepare(self, data, label):
        dim, num = data.shape

        # get H, c, aI, aE
        # H
        H = kernelFunc(data, data, self.kernel)

        yf = np.repeat(label, num)
        yl = np.tile(label, num)
        H = H*((yf*yl).reshape(num, num))
        # c
        c = np.ones((num, 1)) * -1
        # aI, bI
        # aI.T@x < bI
        aI = np.diag(np.ones(num)) * -1
        bI = np.zeros((num, 1))
        # aE, bE
        # aE.T@x = bE
        aE = label.reshape((num, 1))
        bE = np.zeros((1, 1))

        return H, c, aI, bI, aE, bE
    
    def trickInit(self, aI, bI, label):
        # make sure:
        # st. aI.T@x <= bI
        # aE.T@x = bE

        # get initial x
        initX = np.zeros((len(label), 1))
        maskP = label == 1
        maskN = label == -1
        para1 = np.sum(maskP)
        para2 = np.sum(maskN)
        initX[maskP, 0] = para2
        initX[maskN, 0] = para1
        # get initial active set
        initASIdx = np.where(aI.T@initX - bI == 0)

        return initX, initASIdx[0]