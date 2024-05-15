import csv
import pandas as pd
unclean = pd.read_csv("jointdf.csv")
data = []
for index, row in unclean.iterrows():
    dw = row['Distilled water']
    if dw == "1000ml":
        data.append(row.to_dict())
    else:
        continue

df = pd.DataFrame.from_dict(data)
nan_value = float("NaN") 
df.replace("", nan_value, inplace=True) 
df.dropna(how='all', axis=1, inplace=True)
df.to_csv('clean.csv')