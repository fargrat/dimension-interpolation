import timeit
from operator import itemgetter

import skimage
from matplotlib import patches
from scipy.spatial import distance

from skimage import data, io as sk, img_as_ubyte
from skimage.draw import circle
from skimage.feature import match_template
from skimage.filters import threshold_mean, threshold_li, threshold_minimum, threshold_otsu
from skimage.morphology import convex_hull_object, convex_hull_image
from skimage.transform import hough_line
from skimage.color import rgb2grey
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import moments_hu, moments

import config
import detect_lines
import img_preprocessor

config = config.get_default_config()
SIDE_LENGTH = config.getint('SIDE_LENGTH')
ARROW_THRESHOLD = config.getfloat('ARROW_THRESHOLD')

templates = []
templates.append(sk.imread(config.get('TEMPLATE_PATH') + 'top.jpg'))
templates.append(sk.imread(config.get('TEMPLATE_PATH') + 'left.jpg'))
templates.append(sk.imread(config.get('TEMPLATE_PATH') + 'down.jpg'))
templates.append(sk.imread(config.get('TEMPLATE_PATH') + 'right.jpg'))

"""
Pfeilspitzen erkennen und dessen Mittelpunkte zurÃ¼ckgeben
"""

def get_arrowheads(img, lines):
    img = rgb2grey(img)
    img = skimage.img_as_ubyte(img)

    fig, ax = plt.subplots()
    fig.tight_layout()

    data = []
    relevant = []
    id_counter = 0

    # Enden alle Geraden betrachten
    for l in lines:
        start, end = l[1], l[2]
        dst_start, dst_end, index_start, index_end = has_arrowhead(l[0], start, end, img, id_counter)
        if (dst_start != -1) and (dst_end != -1):
            data.append((dst_start, dst_end, start, end, index_start, index_end, l[0]))
        id_counter = index_end

    # Pfeilspitzenpaare sortieren nach Gesamtdistanz zur Template Pfeilspitze
    data = [i for i in data if (i[0] > ARROW_THRESHOLD) and (i[1] > ARROW_THRESHOLD)]
    #data = sorted(data, key=lambda x: x[0] + x[1], reverse=True)
    #data = data[0:10]


    # Auswahl auf Pfeilspitzenpaare mit geringstem Abstand zur Template Pfeilspitze beschrÃ¤nken

    for d in data:
        # Bounding Boxen der Pfeilspitzen einzeichnen
        draw_bounding_box(ax, img, d[2], d[3], d[4], d[5], d[6])
        relevant.append((d[2][1], d[2][0]))
        relevant.append((d[3][1], d[3][0]))

    #print(data)
    print(len(data))
    ax.imshow(img, interpolation='bicubic', cmap=plt.cm.gray)
    #plt.show()

    return relevant

def has_arrowhead(angle, start, end, img, counter):

    # Index ZÃ¤hler fÃ¼r Bounding Boxes hochzÃ¤hlen
    index_start = counter + 1
    index_end = counter + 2

    if abs(angle) == 90:
        # Start betrachten fÃ¼r linkes Ende
        # Ausschnitte invertieren fÃ¼r Momentberechnung; mittels Slices korrekten Ausschnitt des Bildes finden
        cropped_horz_start=np.invert(img[int(start[1] - (SIDE_LENGTH / 2)):int(start[1] + (SIDE_LENGTH / 2)), int(start[0] + 3):int(start[0] + SIDE_LENGTH )])
        cropped_horz_end = np.invert(img[int(end[1] - (SIDE_LENGTH / 2)):int(end[1] + (SIDE_LENGTH / 2)), int(end[0] - SIDE_LENGTH):int(end[0]) - 3])
        #sk.imsave("../data/processed/non/cropped" + str(index_start) + ".png", img_as_ubyte(cropped_horz_start))
        #sk.imsave("../data/processed/non/cropped" + str(index_end) + ".png", img_as_ubyte(cropped_horz_end))
        cropped_horz_start = cropped_horz_start > 127
        cropped_horz_end = cropped_horz_end  > 127
        cropped_horz_start = convex_hull_image(cropped_horz_start)
        cropped_horz_end = convex_hull_image(cropped_horz_end)
        cropped_horz_start = img_as_ubyte(cropped_horz_start)
        cropped_horz_end = img_as_ubyte(cropped_horz_end)

        highest_start = 0
        highest_end = 0
        for t in templates:
            result_start = match_template(cropped_horz_start, t)
            result_end = match_template(cropped_horz_end, t)
            result_start = np.max(result_start)
            result_end = np.max(result_end)
            if result_start > highest_start:
                highest_start = result_start
            if result_end > highest_end:
                highest_end = result_end
        # End betrachten fÃ¼r rechtes Ende
    else:
        cropped_vert_start = np.invert(img[int(start[1] - SIDE_LENGTH):int(start[1] - 3), int(start[0] - (SIDE_LENGTH / 2)):int(start[0] + (SIDE_LENGTH / 2))])
        cropped_vert_end = np.invert(img[int(end[1] + 3):int(end[1] + SIDE_LENGTH), int(end[0] - (SIDE_LENGTH / 2)):int(end[0] + (SIDE_LENGTH / 2))])
        #sk.imsave("../data/processed/non/cropped" + str(index_start) + ".png", img_as_ubyte(cropped_vert_start))
        #sk.imsave("../data/processed/non/cropped" + str(index_end) + ".png", img_as_ubyte(cropped_vert_end))
        cropped_vert_start = cropped_vert_start > 127
        cropped_vert_end = cropped_vert_end  > 127
        cropped_vert_start = convex_hull_image(cropped_vert_start)
        cropped_vert_end = convex_hull_image(cropped_vert_end)

        cropped_vert_start = img_as_ubyte(cropped_vert_start)
        cropped_vert_end = img_as_ubyte(cropped_vert_end)
        highest_start = 0
        highest_end = 0
        for t in templates:
            result_start = match_template(cropped_vert_start, t)
            result_end = match_template(cropped_vert_end, t)
            result_start = np.max(result_start)
            result_end = np.max(result_end)
            if result_start > highest_start:
                highest_start = result_start
            if result_end > highest_end:
                highest_end = result_end

    return highest_start, highest_end, index_start, index_end

def draw_bounding_box(ax, img, start, end, index_start, index_end, angle):

    if abs(angle) == 90:
        start_bb = patches.Rectangle((start[0], start[1] - (SIDE_LENGTH / 2)), SIDE_LENGTH, SIDE_LENGTH, linewidth=1,
                                     edgecolor='b', facecolor='none')
        ax.add_patch(start_bb)
        end_bb = patches.Rectangle((end[0] - (SIDE_LENGTH), end[1] - (SIDE_LENGTH / 2)), SIDE_LENGTH, SIDE_LENGTH, linewidth=1,
                                   edgecolor='b', facecolor='none')

        ax.add_patch(end_bb)

        #ax.text(start[0], start[1], str(index_start), fontsize=9)
        #ax.text(end[0], end[1], str(index_end), fontsize=9)
    else:
        start_bb = patches.Rectangle((start[0] - (SIDE_LENGTH / 2), start[1] - (SIDE_LENGTH)), SIDE_LENGTH, SIDE_LENGTH, linewidth=1,
                                     edgecolor='b', facecolor='none')
        ax.add_patch(start_bb)
        end_bb = patches.Rectangle((end[0] - (SIDE_LENGTH / 2), end[1]), SIDE_LENGTH, SIDE_LENGTH, linewidth=1,
                                   edgecolor='b', facecolor='none')
        ax.add_patch(end_bb)

        #ax.text(start[0], start[1], str(index_start), fontsize=9)
        #ax.text(end[0], end[1], str(index_end), fontsize=9)

def main():
    img = img_preprocessor.preprocess(config.get('FOLDER_PATH') + config.get('IMG'))
    lines, _ = detect_lines.get_lines(img)
    get_arrowheads(img, lines)
    plt.show()


if __name__ == '__main__':
    main()