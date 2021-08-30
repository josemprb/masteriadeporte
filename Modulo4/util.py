import os
from collections import defaultdict
from shutil import copy, copytree, rmtree
import cv2
import const
import mahotas as mh
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, classification_report
import logging
# from pandas_profiling import ProfileReport

logger = logging.getLogger(const.Logger.LOG_NAME_LOGGER)
logger.setLevel(const.Logger.LOG_LEVEL_DEBUG)
formatter = logging.Formatter(const.Logger.LOG_SCREEN_FORMATTER)


# Método auxiliar para dividir los datos de Food101 en entrenamiento y test.
def prepare_data(filepath, src, dest):
    logger.debug("Dividir los datos de Food101 en entrenamiento y test")
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')
    for food in classes_images.keys():
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    logger.debug("Copia finalizada!")


# Método auxiliar para crear los conjuntos de datos que utilizamos en la prueba.
def dataset_mini(food_list, src, dest):
    logger.debug("Crear los conjuntos de datos para la prueba que se va a ejecutar")
    if os.path.exists(dest):
        rmtree(dest)
    os.makedirs(dest)
    for food_item in food_list:
        copytree(os.path.join(src, food_item), os.path.join(dest, food_item))
    logger.debug("Copia finalizada!")


def import_data(path):
    # Cargar las imágenes y hacer las transformaciones para contar con datos para entrenar.
    # Se hacen tres transformaciones: calcular histogramas, los momentos de Hu y los parámetros de Haralick.
    logger.debug("Cargar las imágenes y realizar las transformaciones para obtener variables numéricas")
    X = []
    y = []
    for label in os.listdir(path):
        for i in os.listdir(os.path.join(path, label)):
            image_path = os.path.join(path, label, i)
            img = cv2.imread(image_path, const.Image.RGB)
            img = cv2.resize(img, (const.Image.IMG_HEIGHT, const.Image.IMG_WIDTH), interpolation=const.Image.INTER_AREA)
            if img.ndim > 2:
                img_gray = cv2.cvtColor(img, const.Image.RGB2GRAY)
                img_hu = cv2.HuMoments(cv2.moments(img_gray)).flatten()
                img_har = mh.features.haralick(img_gray).mean(axis=0)
                img_hsv = cv2.cvtColor(img, const.Image.RGB2HSV)
                img_hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
                X.append(np.hstack([img_hist, img_har, img_hu]))
                y.append(label)
    return X, y


# def exploratory_data_analysis(X, y):
#     # Exploratory data analysis using Pandas-Profiling library.
#     logger.debug("Análisis Exploratorio de Datos, utilizando la librería pandas-profiling")
#     df_X = pd.DataFrame(X)
#     df_y = pd.DataFrame(y)
#     df = pd.concat([df_X, df_y], axis=1)
#     profile = ProfileReport(df, title="Exploratory Data Analysis using Pandas Profiling", explorative=True, )
#     profile.to_file("eda_report.html")


# Codificación de las etiquetas.
def label_encoder(y_train, y_test):
    logger.debug("Codificación de las etiquetas")
    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)
    return y_train, y_test


# Selección de variables
def feature_selection(X_train, y_train, X_test):
    # Estimamos la información mutua de dos variables aleatorias. Esto mide la dependencia mutua entre las variables de
    # entrada y la variable objetivo.
    logger.debug("Selección de variables")
    mi = mutual_info_classif(X_train, y_train)
    # Seleccionar las variables que tengan una dependencia con la variable objetivo superior a 0.001.
    return pd.DataFrame(X_train).drop(columns=np.where(mi < const.Prep.MIN_MUTUAL_INFO)[0]).values, \
           pd.DataFrame(X_test).drop(columns=np.where(mi < const.Prep.MIN_MUTUAL_INFO)[0]).values


# Estandarizar las variables (Restar la media y dividir por la desviación típica).
def feature_standardization(X_train, y_train, X_test, y_test):
    logger.debug("Estandarización de variables")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.fit_transform(X_test, y_test)
    return X_train, X_test


# Normalizar las variables (En el rango por defecto, entre 0 y 1).
def feature_normalization(X_train, y_train, X_test, y_test):
    logger.debug("Normalización de variables")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.fit_transform(X_test, y_test)
    return X_train, X_test


# Escalar las variables con los valores por defecto.
def feature_scaling(X_train, y_train, X_test, y_test):
    logger.debug("Escalado de variables")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.fit_transform(X_test, y_test)
    return X_train, X_test


def standardize(X_train, y_train, X_test, y_test, cte):
    if cte == const.Prep.FEATURE_STANDARDIZATION:
        return feature_standardization(X_train, y_train, X_test, y_test)
    elif cte == const.Prep.FEATURE_NORMALIZATION:
        return feature_normalization(X_train, y_train, X_test, y_test)
    elif cte == const.Prep.FEATURE_SCALING:
        return feature_scaling(X_train, y_train, X_test, y_test)


# def split(X, y):
#     # Dividir los datos en datos de entrenamiento y de test.
#     logger.debug("Dividir los datos en entrenamiento y test")
#     return train_test_split(X, y, test_size=const.Prep.TEST_SIZE, random_state=const.Prep.RANDOM_STATE)


def random_forest(random_state=None, criterion='gini', max_depth=None, max_features='auto', n_estimators=100):
    logger.debug("Modelo clasificador Random Forest")
    return RandomForestClassifier(random_state=random_state, criterion=criterion, max_depth=max_depth,
                                  max_features=max_features, n_estimators=n_estimators)


def support_vector_machine(random_state=None, C=1.0, gamma='scale', kernel='rbf'):
    logger.debug("Modelo clasificador de máquinas de vectores de soporte (SVM)")
    return SVC(random_state=random_state, C=C, gamma=gamma, kernel=kernel)


def naive_bayes():
    logger.debug("Clasificador bayesiano ingenuo")
    return GaussianNB()


def model(model, X, y):
    logger.debug("Empieza el modelado")
    if model == const.Model.RANDOM_FOREST:
        rfc = random_forest(random_state=const.Model.RANDOM_STATE)
        # Tunear hiperparámetros.
        logger.debug("Empieza el análisis de sensibilidad de hiperparámetros")
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=const.Model.PARAM_GRID_RF, cv=const.Model.CV_RF).fit(X, y)
        logger.debug("Hiperparámetros óptimos: {}".format(CV_rfc.best_params_))
        # Devolver el mejor de todos los modelos del grid.
        return random_forest(random_state=const.Model.RANDOM_STATE,
                             criterion=CV_rfc.best_params_['criterion'],
                             max_depth=CV_rfc.best_params_['max_depth'],
                             max_features=CV_rfc.best_params_['max_features'],
                             n_estimators=CV_rfc.best_params_['n_estimators']).fit(X, y)
    elif model == const.Model.SVM:
        svc = support_vector_machine(random_state=const.Model.RANDOM_STATE)
        # Tunear hiperparámetros.
        logger.debug("Empieza el análisis de sensibilidad de hiperparámetros")
        CV_svc = GridSearchCV(estimator=svc, param_grid=const.Model.PARAM_GRID_SVC, refit=True,
                              verbose=const.Model.VERBOSE_SVC).fit(X, y)
        logger.debug("Hiperparámetros óptimos: {}".format(CV_svc.best_params_))
        # Devolver el mejor de todos los modelos del grid.
        return support_vector_machine(random_state=const.Model.RANDOM_STATE,
                                      C=CV_svc.best_params_['C'],
                                      gamma=CV_svc.best_params_['gamma']).fit(X, y)
    elif model == const.Model.NAIVE_BAYES:
        # No hay hiperparámetros que tunear
        return naive_bayes().fit(X, y)


def evaluate(model_fit, X, y, average_f1, average_auc, multi_class):
    logger.debug("Empieza la evaluación de los datos")
    y_pred = model_fit.predict(X)

    # Valor F1: Media harmónica entre precisión y exhaustividad.
    f1 = f1_score(y, y_pred, average=average_f1)
    logger.info("El valor f1 siguiendo la estrategia {} a: {}".format(average_f1, f1))

    # Matriz de confusión.
    conf_matrix = confusion_matrix(y, y_pred)
    logger.info("La matriz de confusión es: {}".format(conf_matrix))

    # Área por debajo de la curva ROC.
    y_pred_ohe = np.zeros((y_pred.size, y_pred.max() + 1))
    y_pred_ohe[np.arange(y_pred.size), y_pred] = 1
    roc_auc = roc_auc_score(y, y_pred_ohe, average=average_auc, multi_class=multi_class)
    logger.info("El área debajo de la curva ROC siguiendo la estrategia {}, {} es igual a: {}".format(
        average_auc, multi_class, roc_auc))

    # Informe de clasificación.
    clas_report = classification_report(y, y_pred)
    logger.info("El informe de clasificación es: {}".format(clas_report))
