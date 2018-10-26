import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn 机器学习库
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, classification_report


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

    # cross_validation(x, y)
    print("---------x_under_sample---------")
    cross_validation(x_under_sample, y_under_sample)

    return data


def cross_validation(x, y):
    # 交叉验证
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # print("total:", len(x_train + x_test))
    # print("X_train", len(x_train))
    # print("x_test", len(x_test))
    # print("y_train", len(y_train))
    # print("y_test", len(y_test))
    # print("percentage:", len(x_train) / len(x_train + x_test))
    print_k_fold_scores(x_train, y_train)


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


if __name__ == '__main__':
    credit_card_data = pd.read_csv("data/creditcard.csv")
    # preview_data(credit_card_data)
    standardized_data = process_data(credit_card_data)
    # print(standardized_data.head())
