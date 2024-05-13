# For each species in prokdatacomplete.csv, look at the cultures and fill in the maximal ingredient requirement. Form the final --> prok_ing.csv
import pandas as pd
from ast import literal_eval
import json
opendic = open('referencejsons/ingredients_0.json', encoding = 'utf-8') 
dict_data = json.load(opendic)
Names = [i['name'] for i in dict_data]
from alive_progress import alive_bar
mi = pd.read_csv("mediaingredients.csv")
prok = pd.read_csv("prokdatacomplete.csv")
rows, cols = prok.shape
print(rows, cols)
index = mi[mi.columns[0]] #Ingredient name, 1, 1a, 2, ... 
print(index, type(index))
with alive_bar(len(mi.columns)) as bar:
    
    for media in range(1, len(mi.columns)):
            splice = mi[mi.columns[media]]
            name = splice[0]
            splice.fillna(0, inplace = True)
            ingdict = {}
            index = mi[mi.columns[0]] 
            for i in range(0, len(index)):
                ingdict[index[i]] = splice[i]
            newcol = []
            
            for index, row in prok.iterrows():
                    cultures = literal_eval(row['cultures'])
                    level = []
                    for cul in cultures:
                            level.append(float(ingdict[cul]))
                    #print(level)
                    requirement = max(level)
                    newcol.append(requirement)
            if name in Names:
                  match = next((l for l in dict_data if l["name"] == str(name)), None)
                  id = match['id']
            else:
                  id = media 
            prok.insert(2, str(name) + "_" + str(id), newcol, )
            bar()
print(prok)
prok.to_csv('prok_max_dependencies.csv')