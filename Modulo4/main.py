from datetime import datetime
import logging
import const
import time
from steps import DataPreparation as Prep, Modelling as Mod, Evaluation as Val

logger = logging.getLogger(const.Logger.LOG_NAME_LOGGER)
logger.setLevel(const.Logger.LOG_LEVEL_DEBUG)
formatter = logging.Formatter(const.Logger.LOG_SCREEN_FORMATTER)

# Crear handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(const.Logger.LOG_LEVEL_DEBUG)
console_handler.setFormatter(formatter)

# Añadir handlers al logger
logger.addHandler(console_handler)

if __name__ == '__main__':
    logger.info("El proceso comienza a las {}".format(datetime.now()))

    logger.info("Plato I: Se elige el dataset Food101 como conjunto de datos para analizar")

    logger.info("Plato II: Se elige la variable 'Saber los tipos (o categorías) de comidas que ha ingerido Andrés'")

    logger.info("Plato III: Tareas de preparación de datos")
    s3 = time.time()
    X_train, X_test, y_train, y_test = Prep(const.Prep.FEATURE_STANDARDIZATION).run()
    logger.debug("La preparación de datos ha tardado {} segundos en ejecutarse.".format(time.time() - s3))

    logger.info("Plato IV: Tareas de modelado (algoritmos)")
    s4 = time.time()
    model_fit = Mod(const.Model.NAIVE_BAYES, X_train, y_train).run()
    logger.debug("El modelado de datos ha tardado {} segundos en ejecutarse".format(time.time() - s4))

    logger.info("Plato V: Evaluación del modelado")
    s5 = time.time()
    Val(model_fit, X_test, y_test, const.Val.TYPE_STRATEGY_MICRO, const.Val.TYPE_STRATEGY_MACRO,
        const.Val.TYPE_STRATEGY_OVR).run()
    logger.debug("La evaluación de datos ha tardado {} segundos en ejecutarse".format(time.time() - s5))

    logger.info("El proceso termina a las {}".format(datetime.now()))
