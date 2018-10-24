import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(color_codes=True)
# np.random.seed(sum(map(ord, "regression")))
# tips = sns.load_dataset("tips")
# sns.regplot(x="total_bill", y="tip", data=tips)

tips = sns.load_dataset("tips")
# g = sns.FacetGrid(tips, col="time")
# g.map(plt.hist, "tip")

g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=0.7)
g.add_legend()
plt.show()
