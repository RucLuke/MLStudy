import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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
    x_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']
    return data


if __name__ == '__main__':
    credit_card_data = pd.read_csv("data/creditcard.csv")
    # preview_data(credit_card_data)
    standardized_data = process_data(credit_card_data)

    # print(standardized_data.head())
