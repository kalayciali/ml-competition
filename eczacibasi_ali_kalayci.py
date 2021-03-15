import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor

class Sales(object):

    SEED = 1

    def __init__(self):
        FILENAME = 'data.xlsx'
        df = pd.read_excel(FILENAME)
        df["date"] = pd.to_datetime(df["date"], format='%Y-%m')
        df.set_index(['date'], inplace=True)

        self.raw_data = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_train_data(self):
        return (self.X_train, self.y_train)

    def get_test_data(self):
        return (self.X_test, self.y_test)

    def get_monthly_data(self):
        df = self.raw_data
        monthly_group = df.groupby(by=[df.index.month, 'customer', 'item'])
        summed_over_month = monthly_group["order"].sum().to_frame()
        summed_over_month.reset_index(inplace=True)
        # put indexes to clmns again
        return summed_over_month

    def get_seasonal_data(self):
        df = self.raw_data
        MONTH_TO_SEASON = np.array([
            None,
            'DJF', 'DJF',
            'MAM', 'MAM','MAM',
            'JJA', 'JJA','JJA',
            'SON', 'SON', 'SON',
            'DJF'
        ])

        # if func is input to groupby
        # input of func will be index of db
        seasonal_group = df.groupby(by=[
            lambda x: MONTH_TO_SEASON[x.month],
            'customer', 'item',
        ])
        summed_over_season = seasonal_group["order"].sum().to_frame()
        # put indexes to clmns again
        summed_over_season.reset_index(inplace=True)
        return summed_over_season

    def split_to_test_train_for(self, label, test_size):
        # label is 'seasonal' or 'monthly'
        # assuming 'order' is target

        data = self.get_seasonal_data() if label == "seasonal" else self.get_monthly_data()
        target = data['order']
        data.drop(['order', ], axis=1, inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, test_size=test_size, random_state=self.SEED
        )

    def get_model(self):

        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)
        poly = PolynomialFeatures()

        param_grid = {
        }

        gbr = GradientBoostingRegressor(
            n_estimators=1000,
            random_state=self.SEED,
            max_depth=4,
            min_samples_split=5,
            learning_rate=0.1,
        )

        gbr_params = {
            'clf__subsample': [0.5, 1],
        }

        # param_grid.update(gbr_params)

        rfr = RandomForestRegressor(
            random_state=self.SEED,
            n_estimators=1000,
            min_samples_split=5,
            n_jobs=-1,
        )

        rfr_params = {
            'clf__max_depth': [4, 6]
        }

        # param_grid.update(rfr_params)

        mlp = MLPRegressor(
            random_state=self.SEED,
            hidden_layer_sizes=(1000, ),
            max_iter=500,
            learning_rate='invscaling',
            solver='sgd',
        )

        mlp_params = {
            'clf__alpha': [0.05, 0.1]
        }

        # param_grid.update(mlp_params)

        poisson = PoissonRegressor(
            max_iter=1000,
        )

        poisson_params = {
            'clf__alpha': [0.2, 0.4]
        }

        param_grid.update(poisson_params)

        lin_reg = LinearRegression()


        pipe = Pipeline([
            ('one_hot', one_hot),
            ('clf', poisson),
        ])

        search = GridSearchCV(
            pipe,
            param_grid,
            n_jobs=-1,
            scoring='r2',
        )

        return search

sales = Sales()

sales.split_to_test_train_for('monthly', 0.2)
X_train, y_train = sales.get_train_data()
X_test, y_test = sales.get_test_data()

model = sales.get_model()
model.fit(X_train, y_train)
print("Score of model: ", model.score(X_test, y_test))
print("Best params", model.best_params_)

