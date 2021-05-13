import pandas as pd
df = pd.read_csv("./dataset/train_with_summ.csv")
del df["Unnamed: 0"]
df.head()

