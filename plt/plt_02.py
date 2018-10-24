import matplotlib.pyplot as plt
from numpy import arange
import pandas as pd

# 柱形图
reviews = pd.read_csv('fandango_scores.csv')
cols = ["FILM", "RT_user_norm", "Metacritic_user_nom", "IMDB_norm", "Fandango_Ratingvalue", "Fandango_Stars"]
norm_reviews = reviews[cols]
film_name = norm_reviews.values[0, 0]
num_cols = ["RT_user_norm", "Metacritic_user_nom", "IMDB_norm", "Fandango_Ratingvalue", "Fandango_Stars"]
bar_heights = norm_reviews.ix[0, num_cols].values

print(bar_heights)
bar_positions = arange(1, 6)
print(bar_positions)
fig, ax = plt.subplots()
ax.bar(bar_positions, bar_heights, 0.3)
ax.set_xticks(bar_positions)
ax.set_xticklabels(num_cols, rotation=45)

ax.set_xlabel("Rating Source")
ax.set_ylabel("Rating Source")
ax.set_title("Average User Rating For " + film_name)
plt.show()
