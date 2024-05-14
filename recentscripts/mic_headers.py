import pandas as pd
import json
opendic = open('ingredients_0.json', encoding = 'utf-8') 
dict_data = json.load(opendic)

mic = pd.read_csv('mediaingredients_complete.csv')
headers = (mic.columns.values.tolist())
alt_dict = {}
for i in headers[1:]: 
    for dd in dict_data:
        if str(dd['id']) == str(i):
            alt_dict[i] = dd['name']
newheader = []
for i in headers:
    if i in alt_dict.keys():
        newheader.append(alt_dict[i])
    else:
        newheader.append(i)
print(newheader)
data = []
data.append(newheader)
for index, row in mic.iterrows():
    data.append(row)
df = pd.DataFrame(data)
df.to_csv('new_mic.csv')
