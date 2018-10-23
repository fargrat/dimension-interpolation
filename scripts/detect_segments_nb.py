import uuid

from skimage import data, io as sk, img_as_uint, img_as_ubyte
from skimage.color import rgb2grey
from matplotlib import pyplot as plt, patches
import numpy as np
from skimage.filters import threshold_otsu, threshold_mean

import cclabel
import config
import img_preprocessor
import train_bayes
from sklearn.cluster import DBSCAN
from skimage.morphology import dilation
from skimage.measure import label, regionprops
from PIL import Image
import optimize_pil

config = config.get_default_config()
img_path = config.get('IMG_PATH')


"""
Von cclabel.py Ã¼bergebene Pixel werden zusammengefÃ¼gt und in Dictionary abgespeichert
"""
def segment(labels):
    components = {}
    for l in labels:
        x = l[0]
        y = l[1]
        component = labels[x, y]

        if component in components:
            # Berechne Wert fÃ¼r Ecke links-oben von Segment (niedriger X und niedriger Y Wert)
            if components[component][0] > x:  # niedrigster X Wert
                components[component][0] = x
            if components[component][1] > y:  # niedrigster Y Wert
                components[component][1] = y

                # Berechne Wert fÃ¼r Ecke rechts-unten von Segment (hÃ¶chstes X und hÃ¶chstes Y Wert)
            if components[component][2] < x:  # hÃ¶chster Y Wert
                components[component][2] = x
            if components[component][3] < y:  # hÃ¶chster Y Wert
                components[component][3] = y

        else:
            # 0 niedrigstes X
            # 1 niedrigstes Y
            # 2 hÃ¶chstes X
            # 3 hÃ¶chstes Y
            components[component] = [x, y, x, y]

    return components


"""
Berechne Schwarzanteil in Ã¼bergebenem Segmentbereich
"""
def get_black_ratio(img, x_start, y_start, x_end, y_end):
    selection = img[x_start:x_end, y_start:y_end]
    selection = img_as_ubyte(selection)
    selection = selection < 127

    if len(selection) > 0:
        selection = selection.flatten()
        return sum(selection) / len(selection)


    return 0


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
            rect = patches.Rectangle((c[0], c[1]), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #ax.text(c[2], c[3], str(index), fontsize=9)


"""
FÃ¼r jede Ã¼bergebene Komponente werden 3 Attribute berechnet:
    - GrÃ¶ÃŸenverhÃ¤ltnis
    - Schwarzanteil
    - GrÃ¶ÃŸe im Vergleich zum gesamten Bild
Sind diese Werte korrekt ( > 0) werden diese als liste zurÃ¼ckgegeben
"""
def calculate_features(components, img):
    data = []

    for index in components:
        c = components[index]

        selection = img[c[1]:c[3], c[0]:c[2]]
        width = c[2] - c[0]
        height = c[3] - c[1]
        if height == 0 or width == 0:
            continue
        else:
            if width > height:
                aspect_ratio = width / height
            else:
                aspect_ratio = height / width

        black_ratio = get_black_ratio(img, c[1], c[0], c[3], c[2])
        area_ratio = (width * height) / (img.shape[0] * img.shape[1])

        try:
            labels = label(selection)
            labels[labels > 1] = 1
            if labels.shape[0] > 1 and labels.shape[1] > 1:
                props = regionprops(labels)
                # Es gibt nur ein Label, deshalb Zuweisung auf erstes und einziges Element
                props = props[0]
            else: continue
        except ValueError: continue
        except IndexError: continue

        # GÃ¼ltige Werte (jeder Wert > 0) in array abspeichern
        if not (aspect_ratio == 0 and black_ratio == 0 and area_ratio == 0.0 and props.perimeter >= 1):
            data.append((index, aspect_ratio, black_ratio, area_ratio, props.convex_area, props.perimeter, props.solidity))

            path = config.get('RESULT_PATH') + "segments/"
            #sk.imsave(path + str(index) + ".png", selection)

    return data


"""
Debug Methode, um Accuracy und Loss von Model zu plotten. 
Probleme, wie Overfitting oder falsche Parametrisierung kÃ¶nnen hiermit erkannt werden.
"""
def draw_model_statistics(history):
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')


# Clean non relevant components from list

"""
Center points der Komponenten berechnen.
Einzeln oder angehÃ¤ngt an bestehendes Komponenten Array zurÃ¼ckgeben
"""
def filter_components(basic_components, relevant_components):
    filtered = {}
    center_points = {}

    for index in basic_components:
        c = basic_components[index]

        component = relevant_components.get(index)
        if component == 1:
            # Mittelpunkte berechnen und an eigentliche Komponentenliste anhÃ¤ngen
            filtered[index] = [c[0], c[1], c[2], c[3], (c[2] - c[0]) / 2 + c[0], (c[3] - c[1]) / 2 + c[1]]
            center_points[index] = ((c[2] - c[0]) / 2 + c[0], (c[3] - c[1]) / 2 + c[1])

    return filtered, center_points

"""
Umrisse der Komponenten einzeichnen
"""
def draw_borders(components, labels, ax):
    tmp = []

    for index, point in enumerate(components):
        value = components[point]
        label = labels[index]
        # print("Point: ", value, " Label: ", labels[index])

        # Maximale und minimale Werte fÃ¼r Komponenten berechnen
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
            # Tritt eine unerwartete IndexÃ¼berschreitung auf, so werden die Komponenten einzeln hinzugefÃ¼gt
            tmp.insert(label, value)

    # Werte einzeichnen
    for item in tmp:
        width = item[2] - item[0]
        height = item[3] - item[1]
        rect = patches.Rectangle((item[0], item[1]), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    return tmp

"""
Teilkomponenten Ã¼ber Distanz zu Gesamtkomponenten zusammenfÃ¼gen (Ziffern -> Zahlen)
"""
def calculate_clusters(center_points):
    arr = np.array(list(center_points.values()))

    # DBSCAN Clustering auf center points der Komponenten anwenden
    # siehe: http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    db = DBSCAN(eps=config.getint('EPS'), min_samples=config.getint('MIN_SAMPLES')).fit(arr)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Anzahl Cluster fÃ¼r Debugging berechnen
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    #print(labels)
    return labels

"""
Komponenten auf Dateisystem speichern und optimieren
"""
def save_components(component, img):
    for index, c in enumerate(component):
        # Komponentenausschnitt um je 2 Pixel vergrÃ¶ÃŸern, um abgeschnittene RÃ¤nder zu vermeiden.
        cropped = img[c[1] - 2:c[3] + 2, c[0] - 2:c[2] + 2]
        path = "../data/processed/cropped"
        file_name = path + str(index) + ".jpg"

        # Bildpfad an Optimierungsskript weitergeben
        #optimize_pil.resize(cropped, path, str(index))


def apply_nb(components, img):
    # Features aus Komponenten extrahieren und als .csv Trainingsdaten abspeichern
    features = calculate_features(components, img)
    data = {}
    for line in features:
        data[line[0]] = (line[1], line[2], line[3], line[4], line[5], line[6])

    # Maschinelles Lernverfahren anwenden, um relevante Segmente zu klassifizieren
    relevant_components = train_bayes.apply(data)

    return relevant_components


"""
Segmentierungsablauf starten
"""
def init(img):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Zusammenhangskomponenten erkennen
    labels = cclabel.run_arr(img)

    img = rgb2grey(img)

    # Zusammenhangskomponente zusammenfÃ¼hren
    components = segment(labels)

    """
    data = calculate_features(components, img)
    with open('data.csv', 'w') as file:
        for line in data:
            text = str(line[1]) + ";" + str(line[2]) + ";" + str(line[3]) + ";" + str(line[4]) + ";" + str(line[5]) + ";" + str(line[6]) + ";" + str(line[0])
            file.write(text)
            file.write('\n')
    draw_components(components, ax, img, components)
"""
    #draw_components(components, ax, img, components)

    if train_bayes.exists_model():
        print("Modell existiert")
        relevant_components = apply_nb(components, img)
    else:
        print("Es existiert kein Modell.")
        train_bayes.train()
        relevant_components = apply_nb(components, img)


    # Center points und Komponenten mit Center points berechnen
    cleaned_components, center_points = filter_components(components, relevant_components)

    # ZusammengehÃ¶rigkeit der Komponenten berechnen und einzeichnen
    labels = calculate_clusters(center_points)
    borders = draw_borders(cleaned_components, labels, ax)

    # Komponenten abspeichern und optimieren
    #save_components(borders, img)

    sk.imshow(img)
    # plt.show()

    return borders

def main():
    init(img_preprocessor.preprocess(config.get('FOLDER_PATH') + config.get('IMG')))
    plt.show()


def safe_ln(x, minval=0.0000000001):
    return np.log10(x.clip(min=minval))


if __name__ == '__main__':
    main()