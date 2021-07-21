import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class QuaFunc():
    @staticmethod
    def get_value(X):
        return X[0]**2 + X[1]**2
    # In this problem I directly give the Jacobian based on func
    # f(x,y)= 5x^2+8xy-34x+5y^2-38y+74
    # You can get a approximate value using f(x+\delta) - f(x)
    @staticmethod
    def get_Grid(X):
        return np.array([2*X[0], 2*X[1]])

    @staticmethod
    def get_Hessian(X):
        return np.array([[2, 0], [0, 2]])


class ExpFunc():
    @staticmethod
    def get_value(X):
        return X[0] * np.exp(- X[0]**2 - X[1]**2)
    # In this problem I directly give the Jacobian based on func
    # You can get a approximate value using f(x+\delta) - f(x)
    @staticmethod
    def get_Grid(X):
        com = np.exp(- (X[0]**2) - (X[1]**2))
        return np.array([(1-2*X[0]*X[0])*com, -2*X[0]*X[1]*com])

    @staticmethod
    def get_Hessian(X):
        com = np.exp(- X[0]**2 - X[1]**2)
        h11 = (4*(X[0]**3)-6*X[0])*com
        h12 = (4*(X[0]**2)-2)*X[1]*com
        h21 = (4*(X[0]**2)-2)*X[1]*com
        h22 = (4*(X[1]**2)-2)*X[0]*com
        return np.array([[h11, h12], [h21, h22]])


def LM(optFunc, initX, mu, eps, v=4):
    def get_Q(X, J, G, S):
        # X (,dim)
        # J (,dim)
        # G (dim, dim)
        # S (dim, 1)
        return optFunc.get_value(X) + J@S + 0.5*S.T@G@S

    X = initX
    dim = initX.shape[0]
    steps = [initX]
    mus = [mu]
    while True:
        J = optFunc.get_Grid(X)
        print(J)
        if np.all(np.abs(J) < eps):
            break
        G = optFunc.get_Hessian(X)
        H = G + mu*np.eye(dim)
        # update mu
        while (np.any(np.linalg.eigvals(H) <= 0)):
            mu *= v
            H = G + mu*np.eye(dim)
        # compute direction
        S = -np.linalg.inv(H)@J.reshape((dim, 1))
        # update mu
        nextX = X+S.reshape(-1)
        R = (optFunc.get_value(nextX) - optFunc.get_value(X)) / \
            (get_Q(nextX, J, G, S)-get_Q(X, J, G, S))
        if R < 1/v:
            mu *= v
        elif R > 1-1/v:
            mu /= 2
        mus.append(mu)
        # update X
        if R > 0:
            X = nextX
            steps.append(X)
    return X, np.array(steps), np.array(mus)

def visulize(optFunc, extent, steps):
    X1, X2 = np.mgrid[extent[0, 0]:extent[0, 1]:40j, extent[1, 0]:extent[1, 1]:40j]
    Z = optFunc.get_value(np.array([X1.reshape(-1), X2.reshape(-1)]))
    Z = Z.reshape(X1.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

    plt.cla()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.contourf(X1, X2, Z, 20, cmap='rainbow')
    for f, l in zip(steps[:-1], steps[1:]):
        d = l - f
        plt.arrow(f[0], f[1], d[0], d[1], shape='full', lw=0.1,
                  length_includes_head=True, head_width=.05)
    plt.show()


if __name__ == "__main__":
    X, steps, mus = LM(optFunc=QuaFunc, initX=np.array(
        [-2, -2]), mu=1, eps=1e-3)
    visulize(QuaFunc, extent=np.array([[-2, 2], [-2, 2]]), steps=steps)
