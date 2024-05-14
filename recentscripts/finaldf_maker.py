# For each species in prokdatacomplete.csv, look at the cultures and fill in the maximal ingredient requirement. Form the final --> prok_ing.csv
import pandas as pd
from ast import literal_eval
import json
opendic = open('ingredients_0.json', encoding = 'utf-8') 
dict_data = json.load(opendic)
Names = [i['name'] for i in dict_data]
from alive_progress import alive_bar
mi = pd.read_csv("new_mic.csv")
list_of_mi = mi.to_dict('records')
whole_dict = {}
for lim in list_of_mi:
      whole_dict[lim['MediaID']] = lim

prok = pd.read_csv("no_nan_Ing.csv")
rows, cols = prok.shape
print(rows, cols)
data = []
with alive_bar(len(mi.columns)) as bar:
    
   for index, row in prok.iterrows():
            entrydict = row.to_dict()
            spec_cul = literal_eval(row['cultures'])
            for i in spec_cul:
                  item = whole_dict[str(i)]
                  overall = entrydict | item
                  data.append(overall)
            bar()
df = pd.DataFrame(data)

df.to_csv('jointdf.csv')