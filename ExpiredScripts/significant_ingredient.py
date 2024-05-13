import json
import csv
import pandas as pd
from scipy import stats
# Most up to date version of this dictionary. 
opendic = open('referencejsons/ingredients_0.json', encoding = 'utf-8') 
dict_data = json.load(opendic)
Names = [i['name'] for i in dict_data]
# Used to demonstrate that the program is working. They can be removed in place of actual data.
file = open('correl/protein_coding_genes.txt', 'r')
data = file.readlines()
sigpos = []
signeg = []
for i in range(0, len(data)):
    dictlist = data[i].split(',')
    dictlist = [dl.strip('()') for dl in dictlist]
    name = str(dictlist[0][16:]).strip("'")
    name = name.split()
    if len(name) == 2:
        name = name[0]
    else:
        name = name[:-1]
        name = " ".join(name)
    if float(dictlist[3][8:]) < 0.05 and float(dictlist[2][40:])> 0  :
        sigpos.append([name,dictlist[2][40:],dictlist[3][8:]])
    elif float(dictlist[3][8:]) < 0.05 and float(dictlist[2][40:])< 0:
        signeg.append([name,dictlist[2][40:],dictlist[3][8:]])
    else:
        continue

print(sigpos)
print(signeg)
pos = []
neg = []
with open('sig_genecount.csv', 'a+') as sg:
    w = csv.writer(sg)
    
            
    for i in sigpos:
            if i[0] in Names:
                match = next((l for l in dict_data if l["name"] == str(i[0])), None)
                if type(match['mass']) != float:
                    vals = {'name': i[0], 'id': match['id'], 'correlation': i[1], 'significance': i[2], 'mass': 'unknown', 'type': 'Pos'}
                else:
                    vals = {'name': i[0], 'id': match['id'], 'correlation': i[1], 'significance': i[2], 'mass': match['mass'], 'type': 'Pos'}
                    pos.append(match['mass'])
                w.writerow(vals.values())
    for i in signeg:
        if i[0] in Names:
            match = next((l for l in dict_data if l['name'] == i[0]), None)
            if type(match['mass']) != float:
                vals = {'name': i[0], 'id': match['id'], 'correlation': i[1], 'significance': i[2], 'mass': 'unknown', 'type': 'Neg'}
            else:

                vals = {'name': i[0], 'id': match['id'], 'correlation': i[1], 'significance': i[2], 'mass': match['mass'], 'type': 'Neg'}
                neg.append(match['mass'])
            w.writerow(vals.values())
sig_genecount = pd.read_csv("sig_genecount.csv")
sig_genecount.to_csv("sig_genecount.csv")           

result = stats.kruskal(pos, neg)
print(result)
result2 = stats.f_oneway(pos, neg)
print(result2)