#Construct basic intrinsic values plots against number of media which a species can be cultured on
import pandas as pd
df = pd.read_csv("Objective2/NWidth.csv")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from scipy.optimize import curve_fit
#Colours
#Emerald green  = Protein Coding Genes = #009B77
# Ruby = Gene Counts = #e0115f
# Saphire Blue = Genome Size = #0F52BA

def Drawer(Factor):
#Drawing scatter plots (outliers included)
    FactorasString = " ".join(re.findall('[A-Z][^A-Z]*', Factor))
    coldict = {'ProteinCodingGenes': '#009B77', 'GeneCounts': '#e0115f', 'GenomeSize': '#0F52BA'}
    col = coldict[Factor]
    xarray = np.log10(df[Factor])
    yarray = np.log10(df['NicheWidth'])
    newdf = pd.DataFrame(zip(xarray, yarray), columns = ['log10 ('+ FactorasString + ')', 'log10(Niche Width)'])
    #plt.figure(figsize=(10, 8))
    #scatter_plot = sns.scatterplot(x='log10 (Genome Size)', y='log10(Niche Width)', data = newdf, palette='Set1', s=100)
    tolabel = {386585.0: 'E. coli',99287: 'S. enterica~Typhimurium',208964: 'P. aeruginosa',1313.0: 'S. pneumoniae'}
    scatter_plot = sns.lmplot(newdf, x='log10 ('+ FactorasString + ')', y='log10(Niche Width)',fit_reg = True, order = 4, scatter_kws={"s": 20, 'color': col }, line_kws = {'color': "000000"})

    scatter_plot.set_xlabels('log10 ('+ FactorasString + ')', fontdict = {'size': 18})
    scatter_plot.set_ylabels('log10 (Niche Width)', fontdict = {'size': 18})
    plt.title('Effect of ' + FactorasString + ' on niche width', fontdict = {'size': 23})
    a=2
    X = []
    Y = []
    for i in tolabel.keys():
        if a ==4:
            a = 1
        textpos = {'1': 'baseline', '2' : 'center', '3': 'top'}
        va = textpos[str(a)]
        for index, row in df.iterrows():
                id = int(row['TaxID'])
                if id == i:
                    x = np.log10(row[Factor])
                    y = np.log10(row['NicheWidth'])
                    X.append(x)
                    Y.append(y)
                    plt.text(x, y, tolabel[i], fontsize = 12, verticalalignment = va )
                    a += 1 
                else:
                     continue
    plt.scatter(X,Y, c = '#000000', s = 5)
    plt.grid(True)
    plt.show()

Drawer('GeneCounts')
#remove outliers
'''

def removal_box_plot(mydf, column, threshold):
    sns.boxplot(mydf[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
 
    removed_outliers = mydf[mydf[column] <= threshold]
 
    sns.boxplot(removed_outliers[column])
    plt.title(f'Box Plot without Outliers of {column}')
    plt.show()
    return removed_outliers
 
 
threshold_value = 6
 
#no_outliers = removal_box_plot(df, 'Number_of_media', threshold_value)
outlier_indices = np.where(df['Number_of_media'] > 50)
no_outliers_df = df.drop(outlier_indices[0])
plt.scatter(np.log(no_outliers_df['protein_coding_genes']), no_outliers_df['Number_of_media'], c = '#fa8072', s = 10, alpha = 0.5)
plt.xlabel('Log Genome Size')
plt.ylabel('Number of different media')
plt.show()

'''

