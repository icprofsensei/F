import csv
from Bio import Entrez
import pandas as pd
Entrez.email = "dp521@ic.ac.uk"
prok = pd.read_csv("prok_ing.csv")
data = []
for index, row in prok.iterrows():
    try:
        stream = Entrez.efetch(db = "Taxonomy", id = row['taxid'], retmode = "xml")
        records = Entrez.read(stream)
        lineage = (records[0]['Lineage'])
        if lineage[1] == 'Eukaryota':
            continue
        else:
            print(row['taxid'])
            lin_list = lineage.split(';')
            newentry = " ".join(lin_list[2:])
            entrydict = row.to_dict()
            entrydict['lineage'] = newentry
            data.append(entrydict)
    except:
        entrydict = row.to_dict()
        entrydict['lineage'] = ""
        data.append(entrydict)
df = pd.DataFrame.from_dict(data)
df.to_csv('prok_ing_lin.csv')