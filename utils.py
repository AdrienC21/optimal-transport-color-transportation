import numpy as np
import colour
from math import sqrt
import os
import ot
from time import time

if not(os.path.exists("images")):
    os.mkdir("images")
if not(os.path.exists("images/results")):
    os.mkdir("images/results")

r = np.random.RandomState(42)

illuminant_RGB = np.array([0.31270, 0.32900])
illuminant_XYZ = np.array([0.34570, 0.35850])
chromatic_adaptation_transform = 'Bradford'
RGB_to_XYZ_matrix = np.array([[0.41240000, 0.35760000, 0.18050000],
                              [0.21260000, 0.71520000, 0.07220000],
                              [0.01930000, 0.11920000, 0.95050000]])


def sqeuclidean(x1, x2):
    """Euclidean distance squared

    Args:
        x1 (array): point in space
        x2 (array): point in space

    Returns:
        float: euclidean distance squared between x1 and x2
    """
    a = x2 - x1
    return np.dot(a, a)


def euclidean(x1, x2):
    """Euclidean distance

    Args:
        x1 (array): point in space
        x2 (array): point in space

    Returns:
        float: euclidean distance between x1 and x2
    """
    return sqrt(sqeuclidean(x1, x2))


def macadam(x1, x2):
    """Euclidean distance in MacAdam space

    Args:
        x1 (array): point in space
        x2 (array): point in space

    Returns:
        float: euclidean distance between x1 and x2 both projected
        in MacAdam colorimetric space
    """
    RGB1 = x1
    RGB2 = x2
    XYZ1 = colour.RGB_to_XYZ(RGB1, illuminant_RGB, illuminant_XYZ,
                             RGB_to_XYZ_matrix, chromatic_adaptation_transform)
    xy1 = colour.XYZ_to_xy(XYZ1)
    z1 = 1 - xy1[0] - xy1[1]
    XYZ2 = colour.RGB_to_XYZ(RGB2, illuminant_RGB, illuminant_XYZ,
                             RGB_to_XYZ_matrix, chromatic_adaptation_transform)
    xy2 = colour.XYZ_to_xy(XYZ2)
    z2 = 1 - xy2[0] - xy2[1]
    u2 = np.array([xy1[0], xy1[1], z1])
    v2 = np.array([xy2[0], xy2[1], z2])
    return euclidean(u2, v2)


def distances(x1, x2, metric='sqeuclidean'):
    """Compute distances between all the points of x1 and all the points of x2

    Args:
        x1 (array): list of points
        x2 (array): list of points
        metric (str, optional): Metric to use when calculating distances.
        Defaults to 'sqeuclidean'.

    Returns:
        array: Matrix. Coefficient (i, j) is equal to the distance between
        x1[i] and x2[j]
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    l1 = x1.shape[0]
    l2 = x2.shape[0]
    res = np.empty((l1, l2)).astype(np.double)
    if metric == 'macadam':
        for i in range(0, l1):
            for j in range(0, l2):
                res[i, j] = macadam(x1[i], x2[j])
    elif metric == 'euclidean':
        for i in range(0, l1):
            for j in range(0, l2):
                res[i, j] = euclidean(x1[i], x2[j])
    elif metric == 'sqeuclidean':
        for i in range(0, l1):
            for j in range(0, l2):
                res[i, j] = sqeuclidean(x1[i], x2[j])
    return res


def res_to_couplage(res):
    """List of tuple (i, j). When (i, j) is in res, it means that we link
    x1[i] to x2[j] where x1 and x2 are two different sets of points.

    Args:
        res (list): List of tuple

    Returns:
        array: coupling matrix
    """
    n = len(res)
    couplage = np.zeros((n, n)).astype(np.float64)
    for (i, j) in res:
        couplage[i, j] = 1.0
    return couplage


def imageToMatrix(im):
    """Transform an image into a "1 column" matrix : one pixel per line
    I.shape[0] number of lines
    I.shape[1] number of columns (horizontal pixels)
    I.shape[2]=3 (RGB encoding)

    Args:
        im (array): Image

    Returns:
        array: Flatten image
    """
    return im.reshape((im.shape[0] * im.shape[1], im.shape[2]))


def matrixToImage(X, shape):
    """Transform a flatten image to a regular image

    Args:
        X (array): Flatten image
        shape (tuple): Original dimension of the image X

    Returns:
        array: Image
    """
    return X.reshape(shape)


def minmax(im):
    """Cut extra values of an image. Values lower than 0 become 0,
    values greater than one become 1

    Args:
        im (array): image / flatten image

    Returns:
        array: Modified image
    """
    return np.clip(im, 0, 1)


def apply_optimal_transport(X1, Xs, Xt, row, col):
    """Apply optimal transport using different methods on Xs and Xt.
    Apply a linear modification on X1 based on the closest point on
    Xs and its linked point's color on Xt.

    Args:
        X1 (array): Source image
        Xs (array): source pixels
        Xt (array): target pixels
        row (int): vertical dimension of source image
        col (int): horizontal dimension of source image

    Returns:
        array, array, array, array: modified images
    """

    # EMD (Earth Mover Distance)
    Etime = time()
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    transp_Xs_emd2 = ot_emd.transform(Xs=X1)
    Image_emd2 = minmax(matrixToImage(transp_Xs_emd2, (row, col, 3)))
    print("EDMTransport running time : {t}".format(t=round(time() - Etime, 3)))

    # SinkhornTransport
    Stime = time()
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
    Image_sinkhorn = minmax(matrixToImage(transp_Xs_sinkhorn, (row, col, 3)))
    print("SinkhornTransport running time : "
          "{t}".format(t=round(time() - Stime, 3)))

    # Linear Mapping
    MLtime = time()
    ot_mapping_linear = ot.da.MappingTransport(mu=1e0, eta=1e-8, bias=True,
                                               max_iter=20, verbose=True)
    ot_mapping_linear.fit(Xs=Xs, Xt=Xt)
    transp_Xs_mapping_linear = ot_mapping_linear.transform(Xs=X1)
    Image_mapping_linear = minmax(matrixToImage(transp_Xs_mapping_linear,
                                                (row, col, 3)))
    print("Mapping Linear running time : "
          "{t}".format(t=round(time() - MLtime, 3)))

    # Gaussian Mapping
    MGtime = time()
    ot_mapping_gaussian = ot.da.MappingTransport(mu=1e0, eta=1e-2, sigma=1,
                                                 bias=False, max_iter=10,
                                                 verbose=True)
    ot_mapping_gaussian.fit(Xs=Xs, Xt=Xt)
    transp_Xs_mapping_gaussian = ot_mapping_gaussian.transform(Xs=X1)
    Image_mapping_gaussian = minmax(matrixToImage(transp_Xs_mapping_gaussian,
                                                  (row, col, 3)))
    print("Mapping Gaussian running time : "
          "{t}".format(t=round(time() - MGtime, 3)))

    return Image_emd2, Image_sinkhorn, Image_mapping_linear,\
        Image_mapping_gaussian
