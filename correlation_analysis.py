#Creates a correlation tester function which finds the spearman ranked correlation coefficient between the intrinsic values and each ingredient quantity. 
import pandas as pd
from ast import literal_eval
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
import scipy.stats as ss
prok = pd.read_csv("prok_ing.csv")
headers = []
for series_name, series in prok.items():
        item = series_name.split()
        number = item[len(item) -1]
        #print(number.strip('Requirement'))
        headers.append(series_name)
#print(headers)

def correltester(headers, basetitle):
    sigrel = []
    nottouse = ['name','cultures','taxid','sci_name','chromosome_numbers','genome_size','genome_gc_content','gene_counts','protein_coding_genes']
    for h in headers:
          a = ss.spearmanr(prok[h],prok[basetitle] )
          significance = a[1]
          h = list(h)
          title = "".join([letter for letter in h if letter != ','])
          if title in nottouse: 
                continue
          elif any(np.isnan(val) for val in a):
                continue
          else:
                sigrel.append({'comparison': [title, basetitle], 'result': a, 'correl': a[0]})
    sigrel = sorted(sigrel, key = lambda d: d['correl'], reverse = True)
    #rels = [sr['comparison'] for sr in sigrel]
    with open(basetitle +'.txt', 'a+', encoding= 'utf-8') as f:
        for sr in sigrel:
              f.write(str(sr) + '\n')


for item in ['chromosome_numbers', 'genome_gc_content','gene_counts','protein_coding_genes','genome_size']:
      correltester(headers, item)