import util
import const


class DataPreparation:
    def __init__(self, cte):
        self.cte = cte
        self.food_list = const.Config.FOOD_LIST
        self.src_train = const.Config.SRC_TRAIN
        self.dest_train = const.Config.DEST_TRAIN
        self.src_test = const.Config.SRC_TEST
        self.dest_test = const.Config.DEST_TEST
        self.train_meta_path = const.Config.META_TRAIN
        self.train_src_img = const.Config.SRC_IMG_TRAIN
        self.train_dest_img = const.Config.DEST_IMG_TRAIN
        self.test_meta_path = const.Config.META_TEST
        self.test_src_img = const.Config.SRC_IMG_TEST
        self.test_dest_img = const.Config.DEST_IMG_TEST

    def run(self):
        if const.Config.FIRST_PREPARE_DATA:  # Cambiar a 0 después de la primera ejecución
            util.prepare_data(self.train_meta_path, self.train_src_img, self.train_dest_img)
            util.prepare_data(self.test_meta_path, self.test_src_img, self.test_dest_img)
        util.dataset_mini(self.food_list, self.src_train, self.dest_train)
        util.dataset_mini(self.food_list, self.src_test, self.dest_test)
        self.X_train, self.y_train = util.import_data(const.Config.DEST_TRAIN)
        self.X_test, self.y_test = util.import_data(const.Config.DEST_TEST)
        self.y_train, self.y_test = util.label_encoder(self.y_train, self.y_test)
        self.X_train, self.X_test = util.feature_selection(self.X_train, self.y_train, self.X_test)
        self.X_train, self.X_test = util.standardize(self.X_train, self.y_train, self.X_test, self.y_test, self.cte)
        return self.X_train, self.X_test, self.y_train, self.y_test


class Modelling:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def run(self):
        return util.model(self.model, self.X, self.y)


class Evaluation:
    def __init__(self, model_fit, X, y, average_f1, average_auc, multi_class):
        self.model_fit = model_fit
        self.X = X
        self.y = y
        self.average_f1 = average_f1
        self.average_auc = average_auc
        self.multi_class = multi_class

    def run(self):
        return util.evaluate(self.model_fit, self.X, self.y, self.average_f1, self.average_auc, self.multi_class)
