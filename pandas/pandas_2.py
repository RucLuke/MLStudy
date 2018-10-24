import pandas as pd
from pandas import Series

# Series
fandango = pd.read_csv("fandango_score_comparison.csv")
# print(fandango.columns)
series_film = fandango["FILM"]
# print(type(series_film))
# print(series_film[0:5])
series_rt = fandango["RottenTomatoes"]
# print(series_rt[0:5])
film_names = series_film.values
rt_scores = series_rt.values
series_custom = Series(rt_scores, index=film_names)
print(series_custom[['Minions (2015)', 'Leviathan (2014)']])
