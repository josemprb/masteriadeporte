class Logger:
    LOG_NAME_LOGGER = "Logger Mente Maravillosa"
    LOG_SCREEN_FORMATTER = '%(asctime)s - %(levelname)-8s - %(module)-8s - %(funcName)-8s: %(lineno)d - %(message)s'
    LOG_LEVEL_SCREEN_HANDLER = 'DEBUG'


class Config:
    IMG_FOLDER = 'Yummly28K\images27638'
    JSON_FOLDER = 'Yummly28K\metadata27638'
    IMG_NAME = 'img'
    JSON_NAME = 'meta'
    IMG_TYPE = '.jpg'
    JSON_TYPE = '.json'