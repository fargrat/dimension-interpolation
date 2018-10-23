import numpy
from skimage.color import rgb2grey
from skimage.transform import rescale
import matplotlib.pyplot as plt
import config
import imghdr
from skimage import data, io as sk, img_as_ubyte

config = config.get_default_config()
ALLOWED_IMG_FORMATS = ['jpeg', 'jpg', 'bmp', 'png']
MAX_LENGTH = 2000
FULL_PATH = config.get('FOLDER_PATH') + config.get('IMG')

def preprocess(path):
    # Bildformat überprüfen
    format = imghdr.what(path)
    if format in ALLOWED_IMG_FORMATS:
        print(format)
    else:
        raise Exception('Bildformat ungültig')

    img = sk.imread(path)
    width = img.shape[0]
    height = img.shape[1]
    aspect_ratio = 0

    # Zu kleine Bilder verwerfen
    if (width * height) < 1000000:
        raise Exception('Auflösung zu gering')

    # Bilder auf Breite 2500 mit bleibendem Seitenverhältnis skalieren.
    # Gutes Verhältnis zwischen Auflösung und Performance
    if height > width:
         aspect_ratio = height / width
    else:
         aspect_ratio = width / height

    # Zu extreme Seitenverhältnisse filtern
    if aspect_ratio > 3:
        raise Exception('Ungültiges Seitenverhältnis')

    return img_as_ubyte(img)


def main():
    preprocess(FULL_PATH)

if __name__ == '__main__':
    main()
