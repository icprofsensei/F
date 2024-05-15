#Creates a correlation tester function which finds the spearman ranked correlation coefficient between the intrinsic values and each ingredient quantity. 
import pandas as pd
from ast import literal_eval
import os
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
import scipy.stats as ss
prok = pd.read_csv("clean.csv")
headers = []
os.mkdir('./correl')
for series_name, series in prok.items():
        item = series_name.split()
        number = item[len(item) -1]
        headers.append(series_name)
cols = []
for h in headers:
      if h.startswith('Unnamed'):
            continue
      else:
            cols.append(h)
def correltester(cols, basetitle):
    sigrel = []
    nottouse = ['name','cultures','taxid','sci_name','chromosome_numbers','genome_size','genome_gc_content','gene_counts','protein_coding_genes']
    for h in cols:
          data = []
          for item in prok[h]:
                if str(item).endswith('g/l'):
                      data.append(bool(str(item).rstrip('g/l')))
                elif str(item).endswith('g'):
                      data.append(bool(str(item).rstrip('g')))
                elif str(item).endswith('ml'):
                      data.append(bool(str(item).rstrip('ml')))
                else:
                      data.append(0)
          a = ss.spearmanr(data,prok[basetitle] )
          if h in nottouse: 
                continue
          elif any(np.isnan(val) for val in a):
                continue
          else:
                sigrel.append({'comparison': [h, basetitle], 'correl': a[0], 'significance': a[1]})
    sigrel = sorted(sigrel, key = lambda d: d['correl'], reverse = True)
    #rels = [sr['comparison'] for sr in sigrel]
    with open('correl/'+ basetitle +'.txt', 'a+', encoding= 'utf-8') as f:
        for sr in sigrel:
              f.write(str(sr) + '\n')


for item in ['chromosome_numbers', 'genome_gc_content','gene_counts','protein_coding_genes','genome_size']:
      correltester(cols, item)