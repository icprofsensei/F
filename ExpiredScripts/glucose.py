# Example of per ingredient correlation analysis for glucose
import pandas as pd
from ast import literal_eval
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

#Adding the ingredient names to the ingredient ids

'''
with open('ingredients_0.json') as json_file:
    data = json.load(json_file)
    df = pd.read_csv("mediaingredients.csv")
    header = [h for h in df.head(0)]
    names = ['Ingredient name']
    #print(json.dumps(data, indent = 4))
    for i in range(0, len(data)):
        names.append(data[i]['name'])
    print(names)
    '''
df = pd.read_csv("mediaingredients.csv")
glusplice = df[df.columns[5]]
glusplice.fillna(0, inplace = True)
index = df[df.columns[0]]
glucose = {}
for i in range(0, len(index)):
    glucose[index[i]] = glusplice[i]


prok = pd.read_csv("prokdatacomplete.csv")
glc = []
for index, row in prok.iterrows():
    cultures = literal_eval(row['cultures'])
    glucoselevel = []
    for cul in cultures:
            glucoselevel.append(float(glucose[cul]))
    requirement = max(glucoselevel)
    glc.append(requirement)
prok.insert(4, 'Glucose Requirement', glc, True)

print(prok)

plt.scatter(prok['Glucose Requirement'], np.log10(prok['genome_size']), c = '#fa8072', s = 10, alpha = 0.5)
plt.xlabel('Log 10 Genome Size')
plt.ylabel('Glucose Requirement')
plt.show()