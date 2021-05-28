import matplotlib.pyplot as plt
import colour.plotting
from mpl_toolkits.mplot3d import Axes3D
from utils import *


def plot_original_images(I1, I2):
    """Plot the images I1 and I2 side by side

    Args:
        I1 (array): Image
        I2 (array): Image
    """
    plt.figure(1, figsize=(12.8, 6))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns
    plt.imshow(I1)
    plt.axis('off')
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.axis('off')
    plt.title('Image 2')

    plt.tight_layout()

    plt.savefig("images/results/preview.jpg", bbox_inches='tight')


def plot_pixels_distribution(Xs, Xt, nbpixels):
    """Plot the distribution of the pixels that
    will be coupled using optimal transport

    Args:
        Xs (array): nbpixels of source image
        Xt (array): nbpixels of target image
        nbpixels (int): number of random pixels extracted
        from the original images
    """
    figdistrib = plt.figure(2, figsize=(12.8, 10))

    ax1 = figdistrib.add_subplot(121, projection='3d')
    for i1 in range(nbpixels):
        ax1.scatter(Xs[i1][0], Xs[i1][1], Xs[i1][2], color=Xs[i1])
    ax1.set_xlabel('Rouge')
    ax1.set_ylabel('Vert')
    ax1.set_zlabel('Bleu')
    plt.title('Image 1')

    ax2 = figdistrib.add_subplot(122, projection='3d')
    for i2 in range(nbpixels):
        ax2.scatter(Xt[i2][0], Xt[i2][1], Xt[i2][2], color=Xt[i2])
    ax2.set_xlabel('Rouge')
    ax2.set_ylabel('Vert')
    ax2.set_zlabel('Bleu')
    plt.title('Image 2')

    plt.tight_layout()

    plt.savefig("images/results/pixel_distribution.jpg", bbox_inches='tight')


def plot_macadam_space():
    """Plot a slice of the MacAdam colorimetric space
    """
    colour.plotting.chromaticity_diagram_plot_CIE1931(standalone=False)

    colour.plotting.render(standalone=True, limits=(-0.1, 0.9, -0.1, 0.9),
                           x_tighten=True, y_tighten=True)

    RGB = np.array([0.0, 1.0, 0.0])
    XYZ = colour.RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ,
                            RGB_to_XYZ_matrix, chromatic_adaptation_transform)
    xy = colour.XYZ_to_xy(XYZ)
    print(xy)


def plot_optimal_transport_results(I1, I2, Image_emd2, Image_sinkhorn,
                                   Image_mapping_linear,
                                   Image_mapping_gaussian):
    """Plot the new images obtain by applying optimal transport and
    changing the colors

    Args:
        I1 (array): Source image
        I2 (array): Target image
        Image_emd2 (array): Image obtained by using EMD
        Image_sinkhorn (array): Image obtained by using Sinkhorn
        Image_mapping_linear (array): Image obtained by using Linear Mapping
        Image_mapping_gaussian (array): Image obtained by using
        Gaussian Mapping
    """
    plt.figure(4, figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(I1)
    plt.axis('off')
    plt.title('Image 1')

    plt.subplot(2, 3, 2)
    plt.imshow(I2)
    plt.axis('off')
    plt.title('Image 2')

    plt.subplot(2, 3, 3)
    plt.imshow(Image_emd2)
    plt.axis('off')
    plt.title('EMD (Earth Mover Distance)')

    plt.subplot(2, 3, 4)
    plt.imshow(Image_sinkhorn)
    plt.axis('off')
    plt.title('SinkhornTransport')

    plt.subplot(2, 3, 5)
    plt.imshow(Image_mapping_linear)
    plt.axis('off')
    plt.title('Linear Mapping')

    plt.subplot(2, 3, 6)
    plt.imshow(Image_mapping_gaussian)
    plt.axis('off')
    plt.title('Gaussian Mapping')
    plt.tight_layout()

    plt.savefig("images/results/optimal_transport.jpg", bbox_inches='tight')

    plt.show()
