try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Programm zur Größeninterpolation von Bauteildimensionen anhand von Bemaßungspfeilextraktion'
                   'aus technischen Skizzen. ',
    'author': 'Niels Schlunder',
    'url': '',
    'download_url': '',
    'author_email': 'nisch032@stud.fh-dortmund.de',
    'version': '1.0',
    'install_requires': ['h5py',
                         'keras==2.1.6',
                         'matplotlib',
                         'numpy==1.14.5',
                         'pillow',
                         'pyocr',
                         'scikit-image==0.14.0',
                         'scikit-learn==0.19.2',
                         'tensorflow==1.8.0'],
    'packages': '',
    'scripts': [],
    'name': 'schlunderBA'
}

setup(**config)