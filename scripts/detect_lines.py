import timeit
from math import degrees, atan2, sqrt

from skimage import data, io as sk
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
from skimage.draw import circle
from skimage.filters import threshold_li, threshold_adaptive, threshold_mean
from skimage.morphology import dilation, square, erosion
from skimage.transform import probabilistic_hough_line
from skimage import draw
from skimage.transform import hough_line, hough_line_peaks
from skimage import feature
from matplotlib import cm


# Relevante Linien zurÃ¼ckgeben mit [0]: Winkel, [1]: Erster Punkt, [2]: Zweiter Punkt, [3]: Pfeilspitze 1, [4]: Pfeilspitze 2
import config
import img_preprocessor

config = config.get_default_config()


def get_lines(img):
    img = rgb2grey(img)
    thresh = threshold_mean(img)
    img = img <= thresh

    # http://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
    fig, axes = plt.subplots(1, 2)
    ax = axes.ravel()

    ax[0].imshow(img)

    ax[1].imshow(img, cmap=cm.gray)


    # Linien mittels Probalistischer-Hough-Transformation erkennen
    lines = probabilistic_hough_line(img, threshold=config.getint('HOUGH_THRESHOLD'), line_length=config.getint('HOUGH_LINE_LENGTH'),
                                     line_gap=config.getint('HOUGH_LINE_GAP'))


    # Erlaubte Winkel definieren
    allowed_values = [-180, -90, 90, 180]

    relevant_lines = []
    relevant_lines_no_angle = []
    ax[1].imshow(img * 0)
    for line in lines:
        p0, p1 = line

        # Winkel der Geraden berechnen
        # http://wikicode.wikidot.com/get-angle-of-line-between-two-points
        angle = degrees(atan2((p1[0] - p0[0]), (p1[1] - p0[1])))

        # Nur Winkel mit erlaubten Winkeln darstellen
        if angle in allowed_values:
            relevant_lines.append((angle, p0, p1, list(), list()))
            relevant_lines_no_angle.append((p0, p1))

            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
            ax[1].text(p0[0], p0[1], "P0", fontsize=10)
            ax[1].text(p1[0], p1[1], "P1", fontsize=10)

    return relevant_lines, relevant_lines_no_angle

def main():
    img = img_preprocessor.get_img()
    get_lines(img)
    #plt.show()

if __name__ == '__main__':
    main()