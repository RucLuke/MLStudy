import matplotlib.pyplot as plt
import seaborn_demo as sns
import numpy as np

# uniform_data = np.random.rand(3, 3)
# print(uniform_data)
# heat_map = sns.heatmap(uniform_data)

flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
ax = sns.heatmap(flights, annot=True, fmt="d", linewidths=0.5, cmap="YlGnBu")
plt.show()
