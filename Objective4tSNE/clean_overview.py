import numpy as np
import pandas as pd
prok = pd.read_csv('clean.csv')
colnames = prok.columns.values
names = []
for index, row in prok.iterrows():
                  data = []
                  list  = row['lineage'].split()
                  names.append(list[1])
mydict = {}
namesset = set(names)
names2 = []
for i in namesset:
        names2.append(i)
for index, i in enumerate(names2):
        mydict[i] = index
print(mydict)
overall = []
for index, row in prok.iterrows():
        data = []
        for currentitem in row:
                if currentitem == row['lineage']:
                        val = mydict[currentitem.split()[1]]
                        data.append(val)
                else:
                        data.append(currentitem)
        overall.append(data)
df = pd.DataFrame(overall, columns = colnames)
