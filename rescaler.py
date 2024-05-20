import pandas as pd
import numpy as np
prok = pd.read_csv('clean.csv')
prok2 = prok.copy()

toreadd = ['taxid','sci_name','cultures','sci_name','chromosome_numbers','genome_size','genome_gc_content', 'protein_coding_genes', 'gene_counts', 'lineage', '0', 'complex_medium', 'MediaID','Distilled water']
minidf = pd.DataFrame()
for i in toreadd:
       minidf[i] = prok2[i]
#print(minidf)
for col in prok.columns:
        if col in toreadd:
            del prok[col]
        elif col.startswith('Unnamed'):
            del prok[col]
        else:
            continue
cols = prok.columns.values
overall = []

for index, row in prok.iterrows():
        data = []
        for item in row:
            if str(item).endswith('g/l'):
                    data.append(np.log(float(str(item).rstrip('g/l'))))
            elif str(item).endswith('g'):
                    data.append(np.log(float(str(item).rstrip('g'))))
            elif str(item).endswith('ml'):
                    data.append(np.log(float(str(item).rstrip('ml'))))
            elif item == 0:
                    data.append(-1e4) # log(0)
            elif type(item) == float or type(item) == int:
                    data.append(np.log(item))
            else:
                    data.append(-1e4) # log(0)
        overall.append(data)
df = pd.DataFrame(overall, columns =list(cols) )
df= df.fillna(0)
overall2= []
for index, row in df.iterrows():
        data = []
        total = sum([r for r in row])
        print(total)
        for item in row:
               data.append(float(item/total))
        overall2.append(data)
df2 = pd.DataFrame(overall2, columns =list(cols) )
df_merged = pd.concat([df2, minidf])
df_merged.to_csv('Reprop_log.csv')