# Dimension interpolation in engineering drawings
As a result of my bachelors thesis at Fachhochschule Dortmund, a programm has been developed to read out dimensions from two-dimensional, scanned engineering drawings. The relevant information constist of height and width of assembly components. Several methods of digital image processing and machine learning are used to identify text segments, arrowheads and straight lines. Based on relevative position of arrowheads and lines to each other as well as of arrows and text segments, relevant information can be identified and read out.



## Installation

- Install tesseract (https://github.com/tesseract-ocr/tesseract)

- Install setuptools (https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras)

- Build and install

```
 python3 setup.py build
 sudo python3 setup.py install
```

## Usage

```
sudo ./main
```



## Configuration

The `config.ini` file is used to set all needed paths and enable quick parameter changes. It contains default values for most parameters which have been proved to deliever good results. 

- All parameters ending in `_PATH` must be set and be valid folder paths.

- `IS_BATCH_MODE` allows to enable or disable 
- `MEASURE_RUNTIME` outputs the runtime needed for the execution of either a single image or the total time for batch mode.
- `CLASSIFICATION_METHOD` decides whether to use the `NB` (Naive-Bayes) or `CNN` (Convolutional Neural Network) method to classify relevant segments.
- `IMG` image to analyze. Must be a valid image file inside the `FOLDER_PATH`.
- `LABELED` NB dataset used for classification. Must be a valid labeled dataset inside the `LABEL_PATH`.
