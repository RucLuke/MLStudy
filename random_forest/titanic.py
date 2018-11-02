import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np


def process_data(data):
    """
    数据预处理
    :param data:
    :return:
    """
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2


def predict():
    """
    预测数据
    :return:
    """
    prediction = np.concatenate(predictions, axis=0)
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    accuracy = sum(prediction == titanic_data['Survived']) / len(prediction)
    print(accuracy)


def random_forest():
    """
    随机森林
    :return:
    """
    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=6, min_samples_leaf=1)
    kf = KFold(n_splits=3, random_state=1)
    scores = cross_val_score(alg, titanic_data[predictors], titanic_data["Survived"], cv=kf)
    print(scores.mean())


def linear_regression():
    """
    线性回归
    :return:
    """
    alg = LinearRegression()
    kf = KFold(n_splits=3, random_state=1)
    predictions = []
    for train, test in kf.split(titanic_data):
        train_predictors = (titanic_data[predictors].iloc[train, :])
        train_target = titanic_data["Survived"].iloc[train]
        alg.fit(train_predictors.values, train_target.values.ravel())
        test_predictors = alg.predict(titanic_data[predictors].iloc[test, :])
        predictions.append(test_predictors)

    predict()


def grid_search():
    # alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
    kf = KFold(n_splits=3, random_state=1)
    # scores = cross_val_score(alg, titanic_data[predictors], titanic_data["Survived"], cv=kf)
    # print(scores.mean())
    tree_param_grid = {"min_samples_split": list((3, 6, 9)), "n_estimators": list((10, 50, 100))}
    grid = GridSearchCV(RandomForestClassifier(), param_grid=tree_param_grid, cv=kf)
    grid.fit(titanic_data[predictors], titanic_data["Survived"])
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print(grid.best_params_, grid.best_score_)


if __name__ == '__main__':
    titanic_data = pd.read_csv("titanic_train.csv")
    process_data(titanic_data)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # random_forest()
    grid_search()
