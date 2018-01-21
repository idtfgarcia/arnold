__author__ = 'Ivan'
import numpy as np

def easySurfacePlot(matrix, xlabel = 'X Label', ylabel = 'Y Label', zlabel = 'Z Label'):

    assert type(matrix) is np.ndarray, "MATRIX is not a np.array: %r" % matrix

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, np.shape(matrix)[0], 1)
    y = np.arange(0, np.shape(matrix)[1], 1)
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, matrix)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def matrixPcolorPlot(matrix, xlabel = 'X Label', ylabel = 'Y Label', zlabel = 'Z Label'):

    assert type(matrix) is np.ndarray, "MATRIX is not a np.array: %r" % matrix

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    x = np.arange(0, np.shape(matrix)[0], 1)
    y = np.arange(0, np.shape(matrix)[1], 1)
    X, Y = np.meshgrid(x, y)

    z_min = matrix.min()
    z_max = matrix.max()
    plt.title('pcolor')
    plt.pcolor(X, Y, matrix, cmap='RdBu', vmin=z_min, vmax=z_max)
    plt.title('image (interp. nearest)')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.colorbar()
    plt.show()

def easySeriesPlot(matrix, xlabel = 'X Label', ylabel = 'Y Label', zlabel = 'Z Label'):
    import matplotlib.pyplot as plt

def knn4vector(vector, ensemble, n, p=2.0):

    import pandas as pd

    p = float(p)

    df = pd.DataFrame([range(0,n,1),np.zeros(n)]).transpose();
    df.columns = ['id','distance']

    for iEns in range(0, ensemble.shape[1]):

        # How far away are we from this ensemble member:
        d = np.abs(ensemble[:, [iEns]] - vector)
        d =  np.power( np.sum( np.power(d,p) ),1.0/p)
        if iEns < n:
            df.id[iEns] = iEns
            df.distance[iEns] = d
            maxID = df.distance.idxmax()

        elif d < df.distance[maxID]:
            df.id[maxID] = iEns
            df.distance[maxID] = d
            maxID = df.distance.idxmax()

    return df





