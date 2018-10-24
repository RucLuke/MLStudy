import pandas as pd
import numpy as np

food_info = pd.read_csv("food_info.csv")
# print(type(food_info))
# print(food_info.dtypes)
# print(help(pd.read_csv))

# head 默认显示前5条数据
# print(food_info.head())

# print(food_info.head(3))
# print(food_info.tail(4))
# print(food_info.columns)
# print(food_info.shape)
# print(food_info.loc[0])
# print(food_info.loc[1])
# print(food_info.loc[3:6])

# 打印某一列
# ndb_col = food_info["NDB_No"]
# print(ndb_col)
# 打印某两列
# columns = ["Lipid_Tot_(g)", "Ash_(g)"]
# print(food_info[columns])
# 查找以('g') 结尾的数据
# col_names = food_info.columns.tolist()
# print(col_names)
# gram_columns = []
# for c in col_names:
#     if c.endswith("(g)"):
#         gram_columns.append(c)
#
# gram_df = food_info[gram_columns]
# print(gram_df.head(3))
# print(food_info["Iron_(mg)"])
# div_1000 = food_info["Iron_(mg)"] / 1000
# print(div_1000)
# 排序
# food_info.sort_values("Sodium_(mg)", inplace=True)
# food_info.sort_values("Sodium_(mg)", inplace=True, ascending=False)
# print(food_info["Sodium_(mg)"])
titanic_survival = pd.read_csv("titanic_train.csv")


# print(titanic_survival.head(3))
# age = titanic_survival["Age"]
# age_is_null = pd.isnull(age)
# data_where_age_is_null = age[age_is_null]
#
# good_ages = age[age_is_null == False]
# correct_mean_age = sum(good_ages) / len(good_ages)
# print(correct_mean_age)

# print(age.mean())

# pivot_table 两列之间的关系
# passenger_fare = titanic_survival.pivot_table(index="Pclass", values="Fare", aggfunc=np.mean)
# print(passenger_fare)
# passenger_age = titanic_survival.pivot_table(index="Pclass", values="Age")
# print(passenger_age)
# passenger_stas = titanic_survival.pivot_table(index="Embarked", values=["Fare", "Survived"], aggfunc=np.sum)
# print(passenger_stas)
# drop_na_columns = titanic_survival.dropna(axis=0, subset=["Age", "Sex"])
# print(drop_na_columns)
def hundredth_row(column):
    return column.loc[99]


def not_null_count(column):
    column_null = pd.isnull(column)
    return len(column[column_null])


def which_class(row):
    pclass = row["Pclass"]
    if pd.isnull(pclass):
        return "Unknown"
    elif pclass == 1:
        return "First Class"
    elif pclass == 2:
        return "Second Class"
    elif pclass == 3:
        return "Third Class"


rows = titanic_survival.apply(hundredth_row)
column_null_count = titanic_survival.apply(not_null_count)
classes = titanic_survival.apply(which_class, axis=1)
print(classes)
