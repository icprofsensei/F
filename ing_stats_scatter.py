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
for i in range(0, len(data)):
    
    dictlist = data[i].split(',')
    dictlist = [dl.strip('()') for dl in dictlist]
    corr.append(dictlist[2][40:])
    sig.append(dictlist[3][8:])
    if float(dictlist[3][8:]) < 1e-50:
        tolabel.append([str(dictlist[0][16:]).strip("'"),dictlist[2][40:],dictlist[3][8:]])
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
for i in tolabel:
    textpos = {'1': 'baseline', '2' : 'bottom', '3': 'center_baseline'}
    num = random.randint(1,3)
    va = textpos[str(num)]
    plt.text(np.log10(1/float(i[2])), float(i[1]), (''.join([l for l in i[0] if not l.isdigit()])).removesuffix('Requirement'), fontsize = 9, verticalalignment = va )
plt.show()