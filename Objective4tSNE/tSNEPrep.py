import pandas as pd
from tSNE import tSNEProk
class tSNEprep:
       def __init__(self, source, factor, cutoff, groups):
                  self.source = source
                  self.factor = factor
                  self.cutoff = cutoff
                  self.groups = groups
       def prep(self):
            prok = pd.read_csv(self.source)
            nottouse = ['sci_name','cultures','sci_name','chromosome_numbers','genome_size','genome_gc_content', 'protein_coding_genes', 'gene_counts', 'lineage', '0', 'complex_medium', 'MediaID']
            nottouse.remove(self.factor)
            for col in prok.columns:
                  if col in nottouse:
                        del prok[col]
                  elif col.startswith('Unnamed'):
                        del prok[col]
                  else:
                        continue
            overall = []
            for index, row in prok.iterrows():
                  data = []
                  for item in row:
                        if str(item).endswith('g/l'):
                              data.append(float(str(item).rstrip('g/l')))
                        elif str(item).endswith('g'):
                              data.append(float(str(item).rstrip('g')))
                        elif str(item).endswith('ml'):
                              data.append(float(str(item).rstrip('ml')))
                        elif type(item) == float or type(item) == int:
                              data.append(item)
                        else:
                              data.append(0)
                  overall.append(data)
            df = pd.DataFrame(overall, columns =list(prok.columns.values) )
            df = df.fillna(0)
            df.to_csv('Objective4tSNE/tSNE.csv')
            result = tSNEProk(self.factor, self.cutoff, self.groups)
            result.figmaker()