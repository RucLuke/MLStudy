import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("darkgrid")
x = np.random.normal(size=100)
sns.distplot(x, bins=20, kde=True)
plt.show()
