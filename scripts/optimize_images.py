from skimage import data, io as sk
from skimage.color import rgb2grey
from skimage import exposure
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.filters import threshold_li, gaussian, threshold_li
from skimage import util, img_as_uint
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import rescale
from skimage.morphology import dilation, disk, binary_dilation
from skimage.segmentation import mark_boundaries

from PIL import Image
import optimize_pil

from matplotlib import pyplot as plt
import numpy as np

"""
Altes Verfahren um Bilder mit schlechter Auflösung aufzubessern
"""

def optimize_legacy(img):
    # Gamma anpassen
    img = exposure.adjust_gamma(img, 50)

    # Binarisierung
    thresh = threshold_li(img)
    img = img > thresh

    img = denoise_tv_chambolle(img, weight=0.67)
    thresh = threshold_li(img)
    img = img > thresh
    img = dilation(img, disk(2))
    return img


def optimize(img):
    img = gaussian(img)
    thresh = threshold_li(img)
    img = img > thresh

    return img


def binarize(img):
    thresh = threshold_li(img)
    img = img > thresh
    return img


def character_recognition(path):
    im = Image.open(path)

def print_results():
    # image = sk.imread("../data/processed_data/single_digit_raw.png")
    img = sk.imread("../data/raw_data/Example01.jpg")
    img = rgb2grey(img)

    plt.subplot(221)
    sk.imshow(img)

    plt.subplot(222)
    plt.hist(img.ravel())

    plt.subplot(223)
    img = optimize(img)
    #sk.imsave("../data/processed_data/processed.jpg", img_as_uint(img))
    sk.imshow(img)

    plt.subplot(224)
    plt.hist(img.ravel())


# plt.show()


def main():
    print_results()
    character_recognition("../data/processed_data/processed.jpg")


if __name__ == '__main__':
    main()
