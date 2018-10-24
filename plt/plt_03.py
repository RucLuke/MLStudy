import pandas as pd
import matplotlib.pyplot as plt

# 散点图
fig, ax = plt.subplots()
# ax 是轴 fig 是画板
reviews = pd.read_csv('fandango_scores.csv')
cols = ["FILM", "RT_user_norm", "Metacritic_user_nom", "IMDB_norm", "Fandango_Ratingvalue", "Fandango_Stars"]
norm_reviews = reviews[cols]
ax.scatter(norm_reviews["Fandango_Ratingvalue"], norm_reviews["RT_user_norm"])
ax.set_xlabel("Fandango")
ax.set_ylabel("RT_user_norm")
plt.show()
