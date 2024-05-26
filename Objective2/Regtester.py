#Construct basic intrinsic values plots against number of media which a species can be cultured on
import pandas as pd
df = pd.read_csv("Objective2/NWidth.csv")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Colours
#Emerald green  = Protein Coding Genes = #b3eb7a
# Ruby = Gene Counts = #ffa07a
# Saphire Blue = Genome Size = #4dd2ff
#ffb6c1 = Genome GC Content
def Drawer(Factor):
#Drawing scatter plots (outliers included)
    if Factor == 'GCContent':
         FactorasString = 'Genome GC Content'
    else:
         FactorasString = " ".join(re.findall('[A-Z][^A-Z]*', Factor))
    coldict = {'ProteinCodingGenes': '#b3eb7a', 'GeneCounts': '#ffa07a', 'GenomeSize': '#4dd2ff', 'GCContent':'#ffb6c1' }
    col = coldict[Factor]
    xarray = np.log10(df[Factor])
    yarray = np.log10(df['NicheWidth'])
    newdf = pd.DataFrame(zip(xarray, yarray), columns = ['log10 ('+ FactorasString + ')', 'log10(Niche Width)'])
    newdf2 = pd.DataFrame(zip(xarray, yarray), columns = ['head1', 'head2'])
    model = smf.ols(formula = 'head1~head2', data = newdf2)
    results = model.fit()
    intercept_pvalue = results.pvalues['Intercept']
    slope_pvalue = results.pvalues['head2']
    
    print(results.params)
    print(intercept_pvalue, slope_pvalue)
    print(results.tvalues)
    print(results.t_test([1,0]))
    print(results.f_test(np.identity(2)))
    tolabel = {386585.0: 'E. coli',99287: 'S. enterica~Typhimurium',208964: 'P. aeruginosa',1313.0: 'S. pneumoniae'}
    
    scatter_plot = sns.lmplot(newdf, x='log10 ('+ FactorasString + ')', y='log10(Niche Width)',fit_reg = True, order = 1, scatter_kws={"s": 20, 'color': col }, line_kws = {'color': "000000", 'linewidth': 0.7})
    
    scatter_plot.set_xlabels('log10 ('+ FactorasString + ')', fontdict = {'size': 18})
    scatter_plot.set_ylabels('log10 (Niche Width)', fontdict = {'size': 18})
    plt.title('Effect of ' + FactorasString + ' on niche width', fontdict = {'size': 18})
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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

def NoLogDrawer(Factor):
    if Factor == 'GCContent':
            FactorasString = 'Genome GC Content'
    else:
            FactorasString = " ".join(re.findall('[A-Z][^A-Z]*', Factor))
    coldict = {'ProteinCodingGenes': '#b3eb7a', 'GeneCounts': '#ffa07a', 'GenomeSize': '#4dd2ff', 'GCContent':'#ffb6c1' }
    col = coldict[Factor]
    xarray = np.log10(df[Factor])
    yarray = df['NicheWidth']
    newdf = pd.DataFrame(zip(xarray, yarray), columns = ['log10 ('+ FactorasString + ')', 'Niche Width'])
    newdf2 = pd.DataFrame(zip(xarray, yarray), columns = ['head1', 'head2'])
    model = smf.ols(formula = 'head1~head2', data = newdf2)
    results = model.fit()
    intercept_pvalue = results.pvalues['Intercept']
    slope_pvalue = results.pvalues['head2']
    
    print(results.params)
    print(intercept_pvalue, slope_pvalue)
    print(results.tvalues)
    print(results.t_test([1,0]))
    print(results.f_test(np.identity(2)))
    tolabel = {386585.0: 'E. coli',99287: 'S. enterica~Typhimurium',208964: 'P. aeruginosa',1313.0: 'S. pneumoniae'}
    
    scatter_plot = sns.lmplot(newdf, x='log10 ('+ FactorasString + ')', y='Niche Width',fit_reg = True, order = 1, scatter_kws={"s": 20, 'color': col }, line_kws = {'color': "000000", 'linewidth': 0.7})
    
    scatter_plot.set_xlabels('log10 ('+ FactorasString + ')', fontdict = {'size': 18})
    scatter_plot.set_ylabels('Niche Width', fontdict = {'size': 18})
    plt.title('Effect of ' + FactorasString + ' on niche width', fontdict = {'size': 18})
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
                    y = row['NicheWidth']
                    X.append(x)
                    Y.append(y)
                    plt.text(x, y, tolabel[i], fontsize = 12, verticalalignment = va )
                    a += 1 
                else:
                     continue
    plt.scatter(X,Y, c = '#000000', s = 5)
    plt.grid(True)
    plt.show()

Drawer('GCContent')