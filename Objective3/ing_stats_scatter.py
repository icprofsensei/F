#Constructing correlation scatter plots based on correlation.txt files
import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import random
file = open('correl/protein_coding_genes.txt', 'r')
data = file.readlines()
corr = []
sig = []
tolabel = []
for i in data:
    dict = (literal_eval(i))
    corr.append(dict['correl'])
    if dict['significance'] == 0.0:
        sign= 1e-300
    else:
        sign = dict['significance']
    sig.append(sign)
    if float(dict['significance']) < 1e-80:
        tolabel.append([dict['comparison'][0], dict['correl'], sign])
corr = [float(c) for c in corr]
sig = [(1/float(s)) for s in sig]
#print(corr, len(corr))
#print(sig, len(sig))
print(tolabel)
#Emerald green  = Protein Coding Genes = #009B77
# Ruby = Gene Counts = #e0115f
# Saphire Blue = Genome Size = #0F52BA
plt.scatter(np.log10(sig), corr, s = 10,c= '#009B77', alpha = 0.5)
plt.title('Protein Coding Genes')
plt.xlabel('log10(1 / Significance)')
plt.ylabel('Correlation')
plt.axhline(y=0, ls = '-', lw = 0.4 , c= '#000000' )
plt.axvline(x = 1.30102996, ls = ':', c= '#ff0000' )
a = 3
for i in tolabel:
    if a ==4:
        a = 1
    textpos = {'1': 'baseline', '2' : 'center', '3': 'top'}
    va = textpos[str(a)]
    plt.text(np.log10(1/float(i[2])), float(i[1]), i[0], fontsize = 9, verticalalignment = va )
    plt.scatter(np.log10(1/float(i[2])), float(i[1]), i[0], c= '#000000')
    a += 1 
plt.show()