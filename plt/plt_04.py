import matplotlib.pyplot as plt
import pandas as pd

# 频率直方图
fig, ax = plt.subplots()
# ax 是轴 fig 是画板
reviews = pd.read_csv('fandango_scores.csv')
cols = ["FILM", "RT_user_norm", "Metacritic_user_nom", "IMDB_norm", "Fandango_Ratingvalue", "Fandango_Stars"]
norm_reviews = reviews[cols]
# ax.hist(norm_reviews["Fandango_Ratingvalue"])
44
plt.show()
