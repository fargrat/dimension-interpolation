from PIL import Image
import numpy as np
from skimage.filters import gaussian, threshold_mean, threshold_otsu
from skimage.transform import rotate

import config
from scipy import misc as m
from skimage import io as sk
from skimage import img_as_uint
import matplotlib.pyplot as plt
import matplotlib.cm as cm

config = config.get_default_config()

def resize(img, index, component):
    saved = []

    # Numpy-Array in Pillow einlesen
    im = Image.fromarray(np.uint8(img), 'L')
    scale = config.getint('SCALE_FACTOR')
    # im = Image.open("../data/processed/cropped37.jpg").convert("L")
    width, height = im.size
    width *= scale
    height *= scale
    angle = 180

    # Bild um Faktor skalieren und ANTIALIAS als Interpolationsmethode wählen
    im = im.resize((width, height), Image.ANTIALIAS)

    if height > width:
        im = im.rotate(-90, expand=1)
        angle = 90

    img = np.array(im)

    # Bild optimieren
    thresh = threshold_otsu(img)
    img = img > 127

    #print("Saving to: ", config.get('RESULT_PATH') + index + "_optimized.jpg")
    plt.imsave(config.get('RESULT_PATH') + index + "_optimized.png", img, cmap=cm.gray)
    # Optimiertes Bild abspeichern
    saved.append((index + "_optimized.png", component, angle))

    return saved
