import numpy as np


def kernelFunc(x1, x2, name, **kwargs):
    """Vectorized kernel function, compute the kernel value for each the pair (x1_i, x2_j) where x1_i is in x1, x2_j is in x2
    Args:
        x1(ndarray): data set 1, shape (dim, dataset1_size)
        x1(ndarray): data set 2, shape (dim, dataset2_size)
        name(str): kernel function type, 'linear', 'polynomial' and 'gassian' are available
    Return:
        H(ndarray): kernel value, h[i,j] is kernel value of pair (x1_i, x2_j)
    """
    _, num1 = x1.shape
    _, num2 = x2.shape
    xf = np.repeat(x1, num2, axis=1)
    xl = np.tile(x2, num1)

    if name == 'linear':
        H = np.sum(xf*xl, axis=0)
    elif name == 'poly':
        d = 3
        if 'poly_d' in kwargs.keys():
            d = kwargs['poly_d']
        H = (np.sum(xf*xl, axis=0) + 1)**d
    elif name == 'gaussian':
        std = 2
        if 'std' in kwargs.keys():
            std = kwargs['std']
        H = np.exp(-np.sum((xf - xl)**2, axis=0) / (2*(std**2)))
    else:
        raise NotImplementedError

    H = H.reshape((num1, num2))
    return H


"""
min x.T@H@x + c.T@x
st. aI.T@x <= bI
    aE.T@x = bE
x0 is the initial solution
"""

def asm(initX, initASIdx, H, c, aI, bI, aE, bE, endThr, maxIter):
    """Active Set Method (ASM)
    Args:
        initX(float ndarray): initial feasible solution, shape: (d, 1)
        initASIdx(int list): the inital active set that initX satisfies. It is a list whose element with int value `v` represents the `v`-th inequality constraint (aI[:, v], bI[v]) is in active set.
    """
    x = initX
    ASIdx = initASIdx  # active set, apart from all the equality constriants
    dim, _ = x.shape
    iter = 0

    while iter < maxIter:
        # construct active set
        # a (w, n)
        # b (w, 1)
        IWSize = ASIdx.size
        if IWSize == 0:
            a = aE
            b = bE
        else:
            a = np.hstack((aE, aI[:, ASIdx]))
            b = np.vstack((bE, bI[ASIdx]))
        # solve the problem:
        # min x.T@H@x + c.T@x
        # st. a.T@x = b
        _, col = a.shape
        matL = np.vstack((H, a.T))  # (w+n, w)
        matB = np.vstack((a, np.zeros((col, col))))  # (w+n, n)
        left = np.hstack((matL, matB))
        right = np.vstack((-c, b))
        res = np.linalg.solve(left.T, right)
        newX = res[:dim, :]
        lamda = res[dim:, :]
        # get moving vector d
        d = newX - x
        ILamda = lamda[-IWSize:]
        # If have reached the optimum point of current active set
        if np.all(np.abs(d) < endThr):
            # situation 1
            if IWSize == 0:  # any inequality constraint is useless
                return x
            if np.all(ILamda >= 0):  # find the final optimum solution
                return x
            # situation 2
            # remove the most negative inequality constraint
            # final optimum solution won't be on this constraint's border
            # and solution may go along opposite direction of this constraint
            minIdx = np.argmin(ILamda)
            ASIdx = np.delete(ASIdx, minIdx)
        else:
            # solution can still be optimized under current active set (inequality constraints)
            eps = 1
            minIdx = -1
            #
            for i in range(aI.shape[1]):
                if i not in ASIdx:
                    if aI[:, i]@d > 0:
                        tmpEps = (bI[i] - aI[:, i]@x)/(aI[:, i]@d)
                        if tmpEps <= eps:
                            eps = tmpEps
                            minIdx = i
            # update x
            x += eps*d
            if minIdx != -1:
                # situation 3
                # some constraints not in current active set does not stands in the way of this 'move step'
                # add this constraint to active set
                ASIdx = np.append(ASIdx, minIdx)
            # situation 4:
            # any constraints not in current active set does not stands in the way of this 'move step'
            # nothing, just x += eps*d
        iter += 1
    return x


def trickInit(self, aI, bI, label):
    # this is a trick implement of initializing fucntion
    # only works for SVM
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

def generateInitial(self, aI, bI, aE, bE):
        # TODO big-M method
        pass
