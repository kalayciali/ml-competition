import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import PoissonRegressor

class Sales(object):

    SEED = 1
    DATA_FILENAME = 'data.xlsx'
    TEST_IDS_FILENAME = 'id.xlsx'
    OUTPUT_FILENAME = 'ali_kalayci_eczacibasi.csv'

    def __init__(self):
        # input data
        df = pd.read_excel(self.DATA_FILENAME)
        df["date"] = pd.to_datetime(df["date"], format='%Y-%m')
        df.set_index(['date'], inplace=True)

        test_df = pd.read_excel(self.TEST_IDS_FILENAME,
                                usecols=['date', 'customer', 'item'])
        col_names_order = ['date', 'customer', 'item']
        test_df = test_df.reindex(columns=col_names_order)
        test_df["date"] = pd.to_datetime(test_df["date"], format='%Y-%m')
        test_df["date"] = test_df["date"].dt.month
        test_df.rename(columns={"date": "month"}, inplace=True)

        self.raw_data = df
        self.test_input = test_df
        self.cleaned_data = None
        self.model = None

        # to test with splitting 
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_test_input(self):
        return self.test_input

    def get_train_data(self):
        return (self.X_train, self.y_train)

    def get_test_data(self):
        return (self.X_test, self.y_test)

    def get_data_and_target_for(self, label):

        data = self.get_seasonal_data() if label == "seasonal" else self.get_monthly_data()
        print(data.describe())
        target = data['order']
        data.drop(['order', ], axis=1, inplace=True)
        return (data, target)

    def get_monthly_data(self):
        df = self.raw_data
        monthly_group = df.groupby(by=[df.index.year, df.index.month, 'customer', 'item'])
        summed_over_month = monthly_group["order"].sum().to_frame()
        summed_over_month.index.rename(['year', 'month', 'customer', 'item'], inplace=True)
        summed_over_month.reset_index(inplace=True)
        summed_over_month.drop(['year', ], axis=1, inplace=True)
        self.cleaned_data = summed_over_month
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
            df.index.year,
            lambda x: MONTH_TO_SEASON[x.month],
            'customer', 'item',
        ])
        summed_over_season = seasonal_group["order"].sum().to_frame()
        summed_over_season.index.rename(['year', 'season', 'customer', 'item'], inplace=True)
        summed_over_season.reset_index(inplace=True)
        summed_over_season.drop(['year', ], axis=1, inplace=True)
        # put indexes to clmns again
        self.cleaned_data = summed_over_season
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

    def make_4d_plot_of_data(self, data, target):
        data.replace({"A": 1, "B": 2, "C": 3}, inplace=True)
        x = data["month"]
        y = data["customer"]
        z = data["item"]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(x, y, z, c=target, cmap=plt.hot())
        fig.colorbar(img)
        plt.show()
        # plt.savefig("images/data_4d.png")


    def write_to_csv(self, orders):
        header = "id,order\n"
        with open(self.OUTPUT_FILENAME, "w+") as f:
            f.write(header)
            for i, order in enumerate(orders):
                row = f"{i+1},{order:.2f}\n"
                f.write(row)

    def get_model(self):

        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

        param_grid = {
        }

        poisson = PoissonRegressor(
            max_iter=1000,
            alpha=0.2,
        )

        poisson_params = {
            'clf__alpha': [0.2, 0.4]
        }

        # param_grid.update(poisson_params)

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

        self.model = pipe

        return pipe


sales = Sales()

# Methods for testing on train test split
# sales.split_to_test_train_for('monthly', 0.2)
# X_train, y_train = sales.get_train_data()
# X_test, y_test = sales.get_test_data()

# testing without splitting
data, target = sales.get_data_and_target_for("monthly")
sales.make_4d_plot_of_data(data, target)
test_input = sales.get_test_input()

model = sales.get_model()
model.fit(data, target)
calc_target = model.predict(test_input)
sales.write_to_csv(calc_target)

# scoring for train test split
# print("Score of model: ", model.score(X_test, y_test))
# print("Best params", model.best_params_)
# print("RMSE is ", mean_squared_error(y_test, calc_target))

