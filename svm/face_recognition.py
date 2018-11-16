from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns


def train_best_model(x_train, y_train, model):
    """

    :param x_train:
    :param y_train:
    :param model:
    :return:
    """
    param_grid = {'svc__C': [1, 5, 10, 100],
                  'svc__gamma': [0.0001, 0.0005, 0.001]}
    grid = GridSearchCV(model, param_grid)
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    return grid.best_estimator_


def show_result(x_test, y_test, y_fit):
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(faces.target_names[y_fit[i]].split()[-1],
                       color='black' if y_fit[i] == y_test[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
    plt.show()
    # %%

    print(classification_report(y_test, y_fit,
                                target_names=faces.target_names))

    # %% [markdown]
    # * 精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
    # * 召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
    # * F1 = 2*精度*召回率/(精度+召回率)

    # %%

    mat = confusion_matrix(y_test, y_fit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=faces.target_names,
                yticklabels=faces.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

def pca_func():
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight="balanced")
    model = make_pipeline(pca, svc)
    x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=40)
    model = train_best_model(x_train, y_train, model)
    y_fit = model.predict(x_test)
    print(y_fit.shape)
    show_result(x_test, y_test, y_fit)


if __name__ == '__main__':
    faces = fetch_lfw_people(min_faces_per_person=10)
    #
    # print("always excute")
    print(faces.target_names)
    print(faces.images.shape)
    # fig, ax = plt.subplots(3, 5)
    # for i, axi in enumerate(ax.flat):
    #     axi.imshow(faces.images[i], cmap="bone")
    #     axi.set(xticks=[], yticks=[],
    #             xlabel=faces.target_names[faces.target[i]])
    # plt.show()
    pca_func()
