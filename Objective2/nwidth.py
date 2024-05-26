import pandas as pd
import json
check = pd.read_csv('check.csv')
prok = pd.read_csv('oldcsvs/prokdatacomplete.csv')
prokjson = prok.to_json(orient = "records")
prokobj = json.loads(prokjson)
data = []
for index, row in check.iterrows():
    id = int(row['TaxID'])
    item = next(i for i in prokobj if i['taxid'] == id)
    newrow = {'TaxID': row['TaxID'], 'Resources': row['Resources'], 'NicheWidth': row['Niche_Width'], 'GenomeSize': item['genome_size'], 'GeneCounts': item['gene_counts'], 'ProteinCodingGenes': item['protein_coding_genes'], 'GCContent': item['genome_gc_content']}
    data.append(newrow)
df = pd.DataFrame(data)
df.to_csv('NWidth.csv')