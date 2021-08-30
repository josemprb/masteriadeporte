class Logger:
    LOG_NAME_LOGGER = "Logger Mente Maravillosa"
    LOG_SCREEN_FORMATTER = '%(asctime)s - %(levelname)-8s - %(module)-8s - %(funcName)-8s: %(lineno)d - %(message)s'
    LOG_LEVEL_DEBUG = 'DEBUG'
    LOG_LEVEL_INFO = 'INFO'
    LOG_LEVEL_WARNING = 'WARNING'
    LOG_LEVEL_ERROR = 'ERROR'


class Config:
    FOOD_LIST = ['baby_back_ribs', 'caprese_salad', 'carrot_cake']
    SRC_TRAIN = 'train'
    DEST_TRAIN = 'train_mini'
    SRC_TEST = 'test'
    DEST_TEST = 'test_mini'
    META_TRAIN = 'food-101/food-101/meta/train.txt'
    SRC_IMG_TRAIN = 'food-101/food-101/images'
    DEST_IMG_TRAIN = 'train'
    META_TEST = 'food-101/food-101/meta/test.txt'
    SRC_IMG_TEST = 'food-101/food-101/images'
    DEST_IMG_TEST = 'test'
    FIRST_PREPARE_DATA = 0


class Image:
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    RGB2GRAY = 6
    RGB2HSV = 40
    RGB = 4
    INTER_AREA = 3
    N_IMAGES = 100


class Prep:
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    MIN_MUTUAL_INFO = 0.001
    FEATURE_STANDARDIZATION = 'Standardize Variables'
    FEATURE_NORMALIZATION = 'Normalize Variables'
    FEATURE_SCALING = 'Scale Variables'


class Model:
    RANDOM_FOREST = 'Random Forest'
    SVM = 'Support Vector Machine'
    NAIVE_BAYES = 'Naive Bayes'
    RANDOM_STATE = 42
    PARAM_GRID_RF = {
        'n_estimators': [500, 800, 1000],
        'max_features': ['auto', 'log2', 'sqrt'],
        'max_depth': [10, 12, 14],
        'criterion': ['entropy', 'gini']
    }
    PARAM_GRID_SVC = {
        'C': [5, 10, 50],
        'gamma': [0.005, 0.001, 0.0005]
    }
    CV_RF = 5
    VERBOSE_SVC = 2


class Val:
    TYPE_STRATEGY_MICRO = 'micro'
    TYPE_STRATEGY_MACRO = 'macro'
    TYPE_STRATEGY_WEIGHTED = 'weighted'
    TYPE_STRATEGY_OVO = 'ovo'
    TYPE_STRATEGY_OVR = 'ovr'
