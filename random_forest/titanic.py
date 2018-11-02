import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

"""
第一步：从已知数据提取特征
第二步：数值转换
第三步：构建分类器
第四步：多个分类器
第五步：特征选择（选择重要的特征）
"""


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
    data["FamilySize"] = data["SibSp"] + data["Parch"]
    data["NameLength"] = data["Name"].apply(lambda x: len(x))
    titles = data["Name"].apply(get_title)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mme": 8,
                     "Mlle": 8, "Don": 9, "Lady": 10, "Ms": 3, "Sir": 1, "Jonkheer": 0, "Countess": 0, "Capt": 0}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    # print(pd.value_counts(titles))
    data["Title"] = titles


def random_forest():
    """
    随机森林
    :return:
    """
    n_estimators = 50
    min_samples_split = 3
    min_samples_leaf = 3
    alg = RandomForestClassifier(random_state=1, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf)
    kf = KFold(n_splits=3, random_state=1)
    scores = cross_val_score(alg, titanic_data[predictors], titanic_data["Survived"], cv=kf)
    print("n_estimators=%s min_samples_split=%s min_samples_leaf=%s" % (
        n_estimators, min_samples_split, min_samples_leaf))
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

    # 预测数据
    prediction = np.concatenate(predictions, axis=0)
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    accuracy = sum(prediction == titanic_data['Survived']) / len(prediction)
    print(accuracy)


def grid_search():
    kf = KFold(n_splits=3, random_state=1)
    tree_param_grid = {"min_samples_split": list((3, 6, 9)), "n_estimators": list((10, 50, 100)),
                       "min_samples_leaf": list((1, 2, 3))}
    grid = GridSearchCV(RandomForestClassifier(), param_grid=tree_param_grid, cv=kf)
    grid.fit(titanic_data[predictors], titanic_data["Survived"])
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print(grid.best_params_, grid.best_score_)


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def select_feature():
    """
    选最优参数
    :return:
    """
    selector = SelectKBest(f_classif, k=5)
    selector.fit(titanic_data[predictors], titanic_data["Survived"])
    scores = -np.log10(selector.pvalues_)
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation="vertical")
    plt.show()


def ensemble_boosting():
    """
    集成算法
    :return:
    """
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
         ["Pclass", "Sex", "Fare", "Embarked", "FamilySize", "Title", "Age"]],
        [LogisticRegression(random_state=1, solver="liblinear"),
         ["Pclass", "Sex", "Fare", "Embarked", "FamilySize", "Title", "Age"]]
    ]
    kf = KFold(n_splits=3, random_state=1)
    predictions = []
    for train_index, test_index in kf.split(titanic_data):
        train_target = titanic_data["Survived"].iloc[train_index]
        full_test_predictions = []
        for alg, alg_predictor in algorithms:
            alg.fit(titanic_data[alg_predictor].iloc[train_index, :], train_target)
            test_predictions = alg.predict_proba(titanic_data[alg_predictor].iloc[test_index, :].astype(float))[:, 1]
            full_test_predictions.append(test_predictions)
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
        test_predictions[test_predictions > 0.5] = 1
        test_predictions[test_predictions <= 0.5] = 0
        print(test_predictions)
        predictions.append(test_predictions)

    predictions = np.concatenate(predictions, axis=0)
    accuracy = sum(predictions == titanic_data["Survived"]) / len(predictions)
    print(accuracy)


if __name__ == '__main__':
    titanic_data = pd.read_csv("titanic_train.csv")
    process_data(titanic_data)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]
    # random_forest()
    # select_feature()
    # pick only the five best features
    # predictors = ["Pclass", "Sex", "Fare", "Title"]
    # grid_search()
    ensemble_boosting()
