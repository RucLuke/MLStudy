import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    credit_card_data = pd.read_csv("data/creditcard.csv")
    print(len(credit_card_data))
    print(credit_card_data.head(2))
    print(credit_card_data.loc[0, :])

