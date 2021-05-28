from PIL import Image
from parameters import *
from utils import *
from plot_utils import *

imageSource = "images/{img}".format(img=imageSourceName)
imageTarget = "images/{img}".format(img=imageTargetName)
I1 = np.array(Image.open(imageSource)).astype(np.float64) / 256
I2 = np.array(Image.open(imageTarget)).astype(np.float64) / 256
X1 = imageToMatrix(I1)
X2 = imageToMatrix(I2)

# list of integers corresponding to the coordinates of pixels (lines of Xi)
pixelsx1 = r.randint(X1.shape[0], size=(nbpixels, ))
pixelsx2 = r.randint(X2.shape[0], size=(nbpixels, ))

Xs = X1[pixelsx1, :]  # source pixels
Xt = X2[pixelsx2, :]  # target pixels

# extract the shape (useful if I1 is a black&white image that
# we want to colorize)
SI1 = I1.shape
row = SI1[0]
col = SI1[1]

plot_original_images(I1, I2)
plot_pixels_distribution(Xs, Xt, nbpixels)
plot_macadam_space()

Image_emd2, Image_sinkhorn, Image_mapping_linear,\
        Image_mapping_gaussian = apply_optimal_transport(X1, Xs, Xt, row, col)

plot_optimal_transport_results(I1, I2, Image_emd2, Image_sinkhorn,
                               Image_mapping_linear,
                               Image_mapping_gaussian)
