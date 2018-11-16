import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def sin_plot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)


# sns.set()
# sin_plot()
# sns.set_style("darkgrid")
# sns.set_style("dark")
# sns.set_style("white")
# sns.set_style("ticks")
# data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
# sns.boxplot(data=data)
# sns.despine(offset=10)
# sns.palplot(sns.color_palette("hls"), 8)
data = np.random.normal(size=(20, 8)) + np.arange(8) / 2
# sns.boxplot(data=data, palette=sns.color_palette("Paired", 8))
# sns.palplot(sns.color_palette("Blues_r", 8))
sns.palplot(sns.color_palette("cubehelix", 8))
plt.show()
