#Cleaning the data corresponding to species, [culture id, culture id2, ...] in the csv file. 
import csv
import pandas as pd
import json
from alive_progress import alive_bar
from ast import literal_eval
with open('combined_dup_noblank.csv', 'r') as old: 
    
    r = csv.reader(old)
    species = []
    cultures = []
    for index, row in enumerate(r):
        if index != 0:
            species.append(row[3].strip())
            culture = row[2]
            cultures.append(literal_eval(culture))    
    speciesset = set(species)
    speciesset = list(species)
    data = dict.fromkeys(speciesset, "")
    print('currently iterating')
    with alive_bar(len(speciesset)) as bar:
        for i in range(0, len(species)):
            key = species[i]
            old = data[key]
            value = cultures[i]
            str = ''
            for i in value:
                entry = i + ',' 
                str += entry
            data[key] = old + str
            bar()
    #print('data', data)
    print('Generated dictionary')
    for key in data.keys():
        value = data[key]
        value = value.split(',')
        strainset = set(value)
        strainset = list(strainset)
        strainset = [i.strip('[]"') for i in strainset]
        strainset.remove('')
        data[key] = strainset
    #print(data)
    with open('clean.csv', 'a+') as c:
        w = csv.writer(c)
        for d in data.items():

            w.writerow(d)
    print(True)
    
df = pd.read_csv('clean.csv')
df.to_csv('cleaned.csv', index=False)