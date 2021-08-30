import util
import pandas as pd


class DataPreparation:
    def __init__(self):
        self.img_list = []
        self.df_raw = pd.DataFrame()

    def run(self):
        return util.import_data(self.df_raw, self.img_list)


class Modelling:
    def __init__(self):
        self.name = 'Plato IV'

    def run(self):
        return self.name


class Evaluation:
    def __init__(self):
        self.name = 'Plato V'

    def run(self):
        return self.name
