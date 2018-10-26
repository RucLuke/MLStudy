import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# sklearn 机器学习库
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE


def preview_data(data):
    # 初步观察样本的分布规则
    count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
    print(count_classes)
    count_classes.plot(kind='bar')
    plt.title('Fraud class histogram')
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()


def process_data(data):
    # 预处理，标准化数据列
    # reshape 中 -1，的意思是自动补全
    data["normAmount"] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    x = data.loc[:, data.columns != "Class"]
    y = data.loc[:, data.columns == "Class"]

    number_records_fraud = len(data[data.Class == 1])
    fraud_indices = np.array(data[data.Class == 1].index)
    normal_indices = data[data.Class == 0].index
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    # 合并
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    under_sample_data = data.iloc[under_sample_indices, :]

    # 下采样
    x_under_sample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
    y_under_sample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

    print("---------original sample---------")
    # cross_validation(x, y)
    print("---------under sample---------")
    cross_validation(x_under_sample, y_under_sample)

    return data


def cross_validation(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    predict_prob(x_train, x_test, y_train, y_test)


def predict_prob(x_train, x_test, y_train, y_test):
    # best_c = print_k_fold_scores(x_train, y_train)
    best_c = 0.01
    lr = LogisticRegression(C=best_c, penalty="l1", solver="liblinear")
    lr.fit(x_train, y_train.values.ravel())
    y_predict_prob = lr.predict_proba(x_test.values)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure(figsize=(10, 10))
    j = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_predict_prob[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
        np.set_printoptions(precision=2)
        print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix,
                              classes=class_names,
                              title='Confusion matrix > %s' % i)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_k_fold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)

    # Different C parameters
    c_param_range = [0.005, 0.008, 0.01, 0.1, 1, 10, 100]
    results_table = pd.DataFrame(index=range(len(c_param_range), 2),
                                 columns=['C_parameter'],
                                 dtype=np.float32)
    results_table['C_parameter'] = c_param_range
    j = 0
    for c_param in c_param_range:
        recall_acc_array = []
        i = 1
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # create model
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            y_predict_under_sample = lr.predict(x_train_data.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_predict_under_sample)
            recall_acc_array.append(recall_acc)
            results_table.loc[j, "iter %d" % i] = recall_acc

            i += 1
        results_table.loc[j, "Mean recall score"] = np.mean(recall_acc_array)
        j += 1
    print(results_table)
    best_c = results_table.loc[results_table["Mean recall score"].idxmax()]['C_parameter']

    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c


def smote_sample(data):
    columns = data.columns
    features_columns = columns.delete(len(columns) - 1)
    features = data[features_columns]
    labels = data["Class"]
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=0)
    over_sampler = SMOTE(random_state=0)
    os_features, os_labels = over_sampler.fit_sample(features_train, labels_train)
    print(len(os_labels[os_labels == 1]))
    os_features = pd.DataFrame(os_features)
    os_labels = pd.DataFrame(os_labels)
    # best_c = print_k_fold_scores(os_features, os_labels)
    best_c = 100
    lr = LogisticRegression(C=best_c, penalty="l1",solver="liblinear")
    lr.fit(os_features, os_labels.values.ravel())
    y_predict = lr.predict(features_test.values)
    con_matrix = confusion_matrix(labels_test, y_predict)
    np.set_printoptions(precision=2)
    classes = [0, 1]
    plt.figure()
    plot_confusion_matrix(con_matrix, classes)
    plt.show()


if __name__ == '__main__':
    credit_card_data = pd.read_csv("data/creditcard.csv")
    # preview_data(credit_card_data)
    # standardized_data = process_data(credit_card_data)
    smote_sample(credit_card_data)
    # print(standardized_data.head())
