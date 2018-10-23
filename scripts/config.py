import configparser

CONFIG_FLAVOR = 'TEST'

def get_default_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config[CONFIG_FLAVOR]