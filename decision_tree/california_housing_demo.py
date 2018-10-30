from sklearn.model_selection import train_test_split
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def grid_search():
    tree_param_grid = {"min_samples_split": list((3, 6, 9)), "n_estimators": list((10, 50, 100))}
    grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
    grid.fit(data_train, target_train)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print(grid.best_params_, grid.best_score_)


def random_forest():
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(data_train, target_train)
    rfr_score = rfr.score(data_test, target_test)
    print(rfr_score)


def print_dot_data():
    dot_data = tree.export_graphviz(
        dtr,
        out_file=None,
        feature_names=housing.feature_names[6:8],
        filled=True,
        impurity=False,
        rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    graph.write_png("../pic/tree.png")


if __name__ == '__main__':
    housing = fetch_california_housing()
    data_train, data_test, target_train, target_test = \
        train_test_split(housing.data, housing.target, test_size=0.1, random_state=42)

    dtr = tree.DecisionTreeRegressor(random_state=42)
    dtr.fit(data_train, target_train)
    score = dtr.score(data_test, target_test)
    print(score)
    random_forest()
    grid_search()
    # print_dot_data()
