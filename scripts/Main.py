import glob
import os
import subprocess
import sys
from math import sqrt, hypot
from operator import itemgetter

from PIL import Image
import pyocr
import pyocr.builders
from skimage import data, io as sk
from skimage.draw import circle
from skimage.color import rgb2grey

import config as c
import detect_arrowheads_new
import detect_lines
import detect_segments_cnn
from matplotlib import pyplot as plt, patches
import timeit
import numpy as np

import detect_segments_nb
import img_preprocessor
import optimize_pil
config = c.get_default_config()

RUNS = 1

# Teilzeiten
sub_total = []
sub_arrowhead = []
sub_lines = []
sub_segments = []
sub_combine = []
sub_preprocessing = []


class MyBuilder(pyocr.builders.TextBuilder):
    def __init__(self):
        super().__init__()
        self.tesseract_configs += self.tesseract_configs + ['-l', 'eng','--oem', '3', '--psm', '8']

def main():
    fig, ax = plt.subplots(figsize=(10, 6))

    if config.getboolean('IS_BATCH_MODE'):
        print("Ordnermodus")
        files = _get_all_images()
        print(files)
        if config.getboolean('MEASURE_RUNTIME'):
            for f in files:
                for i in range(0, RUNS):
                    start = timeit.default_timer()
                    img = img_preprocessor.preprocess(config.get('FOLDER_PATH') + "/" + f)
                    stop_preprocessing = timeit.default_timer()
                    sub_preprocessing.append(stop_preprocessing - start)

                    #print("Verarbeite: ", f)
                    process(img, ax)
                    stop = timeit.default_timer()
                    sub_total.append(stop-start)
                    print("Total: ", stop-start)
                    print("Preprocessing: ", stop_preprocessing-start)

                print("_____________________________________NÃ¤chste Bild_____________________________________________________")

        print("Gesamt: ", np.mean(sub_total), " std: ", np.std(sub_total), " alle: ", sub_total)
        print("Arrowheads: ", np.mean(sub_arrowhead), " std: ", np.std(sub_arrowhead), " alle: ", sub_arrowhead)
        print("Lines: ", np.mean(sub_lines), " std: ", np.std(sub_lines), " alle: ", sub_lines)
        print("Segments: ", np.mean(sub_segments), " std: ", np.std(sub_segments), " alle: ", sub_segments)
        print("Preprocessing: ", np.mean(sub_preprocessing), " std: ", np.std(sub_preprocessing), " alle: ", sub_preprocessing)
        print("Combine: ", np.mean(sub_combine), " std: ", np.std(sub_combine), " alle: " , sub_combine)
    else:
        print("Einzelverarbeitungsmodus")
        print("Verarbeite: ", config.get('IMG'))

        if config.getboolean('MEASURE_RUNTIME'):
            for i in range (0, RUNS):
                start = timeit.default_timer()

                img = img_preprocessor.preprocess(config.get('FOLDER_PATH') + config.get('IMG'))
                stop_preprocessing = timeit.default_timer()
                sub_preprocessing.append(stop_preprocessing - start)

                process(img, ax)
                stop = timeit.default_timer()

                sub_total.append(stop-start)

            print("Gesamt: ", np.mean(sub_total), " std: ", np.std(sub_total))
            print("Arrowheads: ", np.mean(sub_arrowhead), " std: ", np.std(sub_arrowhead))
            print("Lines: ", np.mean(sub_lines), " std: ", np.std(sub_lines))
            print("Segments: ", np.mean(sub_segments), " std: ", np.std(sub_segments))
            print("Preprocessing: ", np.mean(sub_preprocessing), " std: ", np.std(sub_preprocessing))
            print("Combine: ", np.mean(sub_combine), " std: ", np.std(sub_combine))
        else:
            img = img_preprocessor.preprocess(config.get('FOLDER_PATH') + config.get('IMG'))
            process(img, ax)

    print("Verarbeitung abgeschlossen.")
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()


def process(img, ax):
    start_lines = timeit.default_timer()
    lines_with_angle, lines = detect_lines.get_lines(img)
    stop_lines = timeit.default_timer()
    sub_lines.append(stop_lines - start_lines)
    print("Sub_lines: ", stop_lines - start_lines)

    start_arrowheads = timeit.default_timer()
    arrowheads = detect_arrowheads_new.get_arrowheads(img, lines_with_angle)
    stop_arrowheads = timeit.default_timer()
    sub_arrowhead.append(stop_arrowheads - start_arrowheads)
    print("Sub_arrowhead: ", stop_arrowheads - start_arrowheads)

    print("Arrowheads: ", arrowheads)
    if(len(lines) == 0 and len(arrowheads) == 0):
        return

    start_segments = timeit.default_timer()
    if config.get('CLASSIFICATION_METHOD') == 'CNN':
        relevant_components = detect_segments_nb.init(img)
    elif config.get('CLASSIFICATION_METHOD') == 'NB':
        relevant_components = detect_segments_nb.init(img)
    else:
        print("UngÃ¼ltiges Klassifikationsverfahren gewÃ¤hlt: ", config.get('CLASSIFICATION_METHOD'))
    stop_segments = timeit.default_timer()
    sub_segments.append(stop_segments - start_segments)
    print("Sub_segments: ", stop_segments - start_segments)

    start_combine = timeit.default_timer()
    relevant_lines = calculate_connected_arrows(arrowheads, lines_with_angle, ax, img)

    relevant = calculate_connected_segments(relevant_lines, relevant_components, ax, img)
    draw_relevant_segments(relevant, ax)

    img = rgb2grey(img)
    r = []
    #print(relevant)
    for item in relevant:
        if item not in r:
            r.append(item)

    save_components(r, img)

    stop_combine = timeit.default_timer()
    sub_combine.append(stop_combine - start_combine)
    print("Sub_combine: ", stop_combine - start_combine)

"""
ZusammengehÃ¶rige Geraden und Segmente durch Mindestabstand ermitteln
"""
def calculate_connected_segments(lines, components, ax, img):
    relevant = []

    for line in lines:
        p0 = line[1]
        p1 = line[2]

        # Mittelpunkte der Geraden bestimmen
        center_x = (p0[0] + p1[0]) / 2
        center_y = (p0[1] + p1[1]) / 2

        # Mittelpunkte fÃ¼r Debugging und Visualisierung einzeichnen
        rr, cc = circle(center_y, center_x, 10)
        #img[rr, cc] = (0, 255, 255)

        # FÃ¼r jede Gerade alle Segmente durchlaufen
        for component in components:
            center = ((component[3] + component[1]) / 2, (component[2] + component[0]) / 2)
            # Mittelpunkt der Segmente einzeichnen
            rr, cc = circle(center[0], center[1], 10)
            #img[rr, cc] = (255, 255, 0)

            # Mittels Pythagoras Abstand zwischen Gerade und Segment berechnen
            distance = hypot(center_x - center[1], center[0] - center_y)

            # Ist die Distanz kleiner als MIN_DISTANCE_SEGMENT_ARROW, wird das Segment als relevant abgespeichert und eine Linie zwischen
            # Gerade und Segment eingezeichnet
            if distance < config.getint('MAX_DISTANCE_SEGMENT_ARROW'):
                relevant.append(component)
                x = [center[1], center_x]
                y = [center[0], center_y]
                ax.plot(x, y)

    return relevant


"""
Geraden mit 2 Pfeilspitzen bestimmen
"""
def calculate_connected_arrows(arrowheads, lines, ax, img):
    relevant_lines = []

    for line in lines:
        p0_relevant = False
        p1_relevant = False

        for arrowhead in arrowheads:
            # Pfeilspitzen einzeichnen
            rr, cc = circle(arrowhead[0], arrowhead[1], 10)
            #img[rr, cc] = (255, 0, 0)

            # Endpunkte der Geraden abspeichern
            p0 = line[1]
            p1 = line[2]

            # Distanz der Pfeilspitze zu beiden Endpunkten der Geraden berechnen
            distance_p0 = hypot(p0[0] - arrowhead[1], p0[1] - arrowhead[0])
            distance_p1 = hypot(p1[0] - arrowhead[1], p1[1] - arrowhead[0])

            # Distanz fÃ¼r beide Enden beschrÃ¤nken
            if distance_p0 < config.getint('MAX_DISTANCE_LINE_ARROWHEAD'):
                # print("Distance p0 under 100")
                p0_relevant = True
            if distance_p1 < config.getint('MAX_DISTANCE_LINE_ARROWHEAD'):
                # print("Distance p0 under 100")
                p1_relevant = True

        # Ist an beiden Enden eine Pfeilspitze mit weniger als MIN_DISTANCE_LINE_ARROWHEAD Pixeln Abstand, wird die Gerade als relevant abgespeichert
        if p0_relevant == True and p1_relevant == True:
            relevant_lines.append(line)
            ax.plot((line[1][0], line[2][0]), (line[1][1], line[2][1]))

    print("Relevant lines: ", relevant_lines)

    return relevant_lines

def draw_relevant_segments(components, ax):
    print(components)
    for c in components:
        # HÃ¶he und Breite berechnen; jeweils 5 Pixel aufaddieren um Abschneiden zu vermeiden
        width = c[2] - c[0] + 5
        height = c[3] - c[1] + 5

        rect = patches.Rectangle((c[0], c[1]), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

def save_components(component, img):
    values_width = []
    values_height = []

    for index, c in enumerate(component):
        # Komponentenausschnitt um je 2 Pixel vergrÃ¶ÃŸern, um abgeschnittene RÃ¤nder zu vermeiden.
        cropped = img[c[1] - 2:c[3] + 2, c[0] - 2:c[2] + 2]
        path = "../data/processed/"
        file_name = path + str(index) + ".jpg"

        # Bildpfad an Optimierungsskript weitergeben
        segments = optimize_pil.resize(cropped, str(index), c)

        for s in segments:
            segment = s[0]
            component = s[1]
            angle = s[2]
            # Ergebnis an Tesseract Ã¼bergeben
            tools = pyocr.get_available_tools()
            if len(tools) == 0:
                print("No OCR tool found")
                sys.exit(1)
            tool = pyocr.tesseract
            builder = MyBuilder()
            ocr_result = tool.image_to_string(Image.open(config.get('RESULT_PATH') + segment), lang='eng', builder=builder)

            # Leerzeichen entfernen
            ocr_result = ocr_result.replace(' ', '')
            print(ocr_result)
            if not ocr_result.startswith("R"):
                # Wenn ocr_result nur Zahlen enthÃ¤lt, konvertiere zu String
                try:
                    ocr_result = float(ocr_result)

                    # Maximale HÃ¶he und Breite bestimmen
                    if angle == 180:
                        values_width.append((ocr_result, component))
                    elif angle == 90:
                        values_height.append((ocr_result, component))

                except ValueError: continue


    max_height = -1
    max_width = -1
    if len(values_height) > 0:
        max_height = max(values_height, key=itemgetter(0))[0]
    if len(values_width) > 0:
        max_width = max(values_width, key=itemgetter(0))[0]

    print("Hoehe: ", max_height, " Breite: ", max_width)

def _get_all_images():
    os.chdir(config.get('FOLDER_PATH'))

    files = []
    for i in img_preprocessor.ALLOWED_IMG_FORMATS:
        f = glob.glob("*." + i)
        if len(f) > 0:
            for element in f:
                files.append(element)
    print(files)

    return files

if __name__ == '__main__':
    main()