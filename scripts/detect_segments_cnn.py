import uuid

from numpy import mean
from scipy import ndimage
from skimage import data, io as sk, img_as_uint
from skimage.color import rgb2grey
from matplotlib import pyplot as plt, patches
import numpy as np
from skimage.filters import threshold_mean, threshold_li, threshold_minimum, threshold_otsu
from skimage.transform import resize
import cclabel
import config
import img_preprocessor
import train_bayes
from sklearn.cluster import DBSCAN
from skimage.morphology import dilation
from skimage.measure import label, moments_central, moments_normalized, moments_hu, regionprops
from PIL import Image
import optimize_pil
import predict_binary

config = config.get_default_config()
img_path = config.get('IMG_PATH')


"""
Von cclabel.py übergebene Pixel werden zusammengefügt und in Dictionary abgespeichert
"""
def segment(labels):
    components = {}
    for l in labels:
        x = l[0]
        y = l[1]
        component = labels[x, y]

        if component in components:
            # Berechne Wert für Ecke links-oben von Segment (niedriger X und niedriger Y Wert)
            if components[component][0] > x:  # niedrigster X Wert
                components[component][0] = x
            if components[component][1] > y:  # niedrigster Y Wert
                components[component][1] = y

                # Berechne Wert für Ecke rechts-unten von Segment (höchstes X und höchstes Y Wert)
            if components[component][2] < x:  # höchster X Wert
                components[component][2] = x
            if components[component][3] < y:  # höchster Y Wert
                components[component][3] = y

        else:
            # 0 niedrigstes X
            # 1 niedrigstes Y
            # 2 höchstes X
            # 3 höchstes Y
            components[component] = [x, y, x, y]

    return component

"""
Debug Methode, um Segmente mit id einzuzeichnen
"""
def draw_components(components, ax, img, relevant_components):
    for index in components:
        c = components[index]

        width = c[2] - c[0]
        height = c[3] - c[1]

        component = relevant_components.get(index)

        if not component is None:
            rect = patches.Rectangle((c[0], c[1]), width, height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.text(c[2], c[3], str(index), fontsize=9)

def calculate_features(components, img):
    data = []
    img = rgb2grey(img)
    thresh = threshold_otsu(img)
    img = img > thresh

    for index in components:
        c = components[index]
        selection = img[c[1]:c[3], c[0]:c[2]]
        delta_left = 32 - selection.shape[0]
        delta_top = 32 - selection.shape[1]
        if (delta_left > 0 and delta_top > 1):
            arr = np.ones(1024).reshape(32, 32)
            selection_width = selection.shape[0] / 2
            selection_height = selection.shape[1] / 2
            arr[int(16 - selection_width):int(16 + selection_width),int(16 - selection_height):int(16 + selection_height)] = selection
            selection = arr
        else:
            selection = resize(selection, (28, 28), anti_aliasing=True)
            delta_left = 32 - selection.shape[0]
            delta_top = 32 - selection.shape[1]
            if delta_left > 0 and delta_top > 1:
                arr = np.ones(1024).reshape(32, 32)
                selection_width = selection.shape[0] / 2
                selection_height = selection.shape[1] / 2
                arr[int(16 - selection_width):int(16 + selection_width),
                int(16 - selection_height):int(16 + selection_height)] = selection
                selection = arr

        name = config.get('IMG').split(".")[0]
        #sk.imsave("../data/processed/segments/" + str(index) + "_" + name + ".png", img_as_uint(selection))

"""
Center points der Komponenten berechnen.
Einzeln oder angehängt an bestehendes Komponenten Array zurückgeben
"""
def filter_components(basic_components, relevant_components):
    filtered = {}
    center_points = {}

    for index in basic_components:
        c = basic_components[index]

        component = relevant_components.get(index)
        if component == 1:
            # Mittelpunkte berechnen und an eigentliche Komponentenliste anhängen
            filtered[index] = [c[0], c[1], c[2], c[3], (c[2] - c[0]) / 2 + c[0], (c[3] - c[1]) / 2 + c[1]]
            center_points[index] = ((c[2] - c[0]) / 2 + c[0], (c[3] - c[1]) / 2 + c[1])

    return filtered, center_points

def calc_center_points(components):
    center_points = {}

    for index in components:
        c = components[index]
        center_points[index] = ((c[2] - c[0]) / 2 + c[0], (c[3] - c[1]) / 2 + c[1])

    return center_points

"""
Umrisse der Komponenten einzeichnen
"""
def draw_borders(components, labels, ax):
    tmp = []

    for index, point in enumerate(components):
        value = components[point]
        label = labels[index]
        # print("Point: ", value, " Label: ", labels[index])

        # Maximale und minimale Werte für Komponenten berechnen
        try:
            saved_value = tmp[label]
            if value[0] < saved_value[0]:
                saved_value[0] = value[0]
            if value[1] < saved_value[1]:
                saved_value[1] = value[1]
            if value[2] > saved_value[2]:
                saved_value[2] = value[2]
            if value[3] > saved_value[3]:
                saved_value[3] = value[3]

        except IndexError:
            # Tritt eine unerwartete Indexüberschreitung auf, so werden die Komponenten einzeln hinzugefügt
            tmp.insert(label, value)

    # Werte einzeichnen
    for item in tmp:
        width = item[2] - item[0]
        height = item[3] - item[1]
        rect = patches.Rectangle((item[0], item[1]), width, height, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    return tmp

"""
Teilkomponenten über Distanz zu Gesamtkomponenten zusammenfügen (Ziffern -> Zahlen)
"""
def calculate_clusters(center_points):
    arr = np.array(list(center_points.values()))

    # DBSCAN Clustering auf center points der Komponenten anwenden
    # siehe: http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    db = DBSCAN(eps=config.getint('EPS'), min_samples=config.getint('MIN_SAMPLES')).fit(arr)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Anzahl Cluster für Debugging berechnen
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    #print(labels)
    return labels

"""
Komponenten auf Dateisystem speichern und optimieren
"""
def save_components(component, img):
    for index, c in enumerate(component):
        # Komponentenausschnitt um je 2 Pixel vergrößern, um abgeschnittene Ränder zu vermeiden.
        cropped = img[c[1] - 2:c[3] + 2, c[0] - 2:c[2] + 2]
        path = "../data/processed/cropped"
        file_name = path + str(index) + ".jpg"

        # Bildpfad an Optimierungsskript weitergeben
        #optimize_pil.resize(cropped, path, str(index))


"""
Segmentierungsablauf starten
"""
def init(img):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Zusammenhangskomponenten erkennen
    labels = cclabel.run_arr(img)

    #img = rgb2grey(img)

    # Zusammenhangskomponente zusammenführen
    components = segment(labels)
    #print(components)

    # Features aus Komponenten extrahieren und als .csv Trainingsdaten abspeichern
    #draw_components(components, ax, img, components)
    data = calculate_features(components, img)
    #print(data[0])
    #with open('data.csv', 'w') as file:
    #    for line in data:
    #        text = str(line[0]) + ";" + str(line[1])
    #        file.write(text)
    #        file.write('\n')

    #calculate_features(components, img)
    relevant_components = {}
    img = rgb2grey(img)
    thresh = threshold_otsu(img)
    img = img > thresh
    for component in components:
        c = components[component]

        selection = img[c[1]:c[3], c[0]:c[2]]
        if selection.shape[0] in range(1, int(img.shape[0] / 10)) and selection.shape[1] in range(1, int(img.shape[1] / 4)):
            if selection.shape == (32, 32, 3):
                selection = selection[:, :, 0]

            delta_left = 32 - selection.shape[0]
            delta_top = 32 - selection.shape[1]
            if (delta_left > 0 and delta_top > 1):
                arr = np.ones(1024).reshape(32, 32)
                selection_width = selection.shape[0] / 2
                selection_height = selection.shape[1] / 2
                arr[int(16 - selection_width):int(16 + selection_width),
                int(16 - selection_height):int(16 + selection_height)] = selection
                selection = arr
            else:
                #print(selection.shape)
                selection = resize(selection, (28, 28))
                arr = np.ones(1024).reshape(32, 32)
                selection_width = selection.shape[0] / 2
                selection_height = selection.shape[1] / 2
                arr[int(16 - selection_width):int(16 + selection_width),
                int(16 - selection_height):int(16 + selection_height)] = selection
                selection = arr

            #sk.imsave("../data/processed/predicted/" + str(component) + ".png",
            #          img_as_uint(selection))

            #print(component)
            type = predict_binary.predict(selection)

            if type == "segment":
                relevant_components[component] = c

    # Center points und Komponenten mit Center points berechnen
    #cleaned_components, center_points = filter_components(components, relevant_components)
    center_points = calc_center_points(relevant_components)

    # Zusammengehörigkeit der Komponenten berechnen und einzeichnen
    labels = calculate_clusters(center_points)
    borders = draw_borders(relevant_components, labels, ax)

    # Komponenten abspeichern und optimieren
    save_components(borders, img)

    sk.imshow(img)
    plt.show(block=False)

    return borders


def main():
    init(img_preprocessor.preprocess(config.get('FOLDER_PATH') + config.get('IMG')))

if __name__ == '__main__':
    main()
