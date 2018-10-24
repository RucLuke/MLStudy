import pandas as pd
import matplotlib.pyplot as plt

un_rate = pd.read_csv("unrate.csv")
un_rate["DATE"] = pd.to_datetime(un_rate["DATE"])
# first_twelve = un_rate[0:12]
#
# plt.plot(first_twelve['DATE'], first_twelve["VALUE"])
# plt.xticks(rotation=45)
# plt.xlabel("Month")
# plt.ylabel("Unemployment rate")
# plt.title("Monthly Unemployment Trends, 1948")

# fig = plt.figure(figsize=(3, 3))
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)

un_rate["MONTH"] = un_rate["DATE"].dt.month
un_rate["MONTH"] = un_rate["DATE"].dt.month
fig = plt.figure(figsize=(6, 3))

plt.plot(un_rate[0:12]["MONTH"], un_rate[0:12]['VALUE'], c='red', label='red')
plt.plot(un_rate[12:24]["MONTH"], un_rate[12:24]['VALUE'], c='blue', label='blue')
plt.legend(loc="best")
plt.show()
