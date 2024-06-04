#Construct basic intrinsic values plots against number of media which a species can be cultured on
import pandas as pd
df = pd.read_csv("Objective2/NWidth.csv")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures

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
    
    x = np.array(newdf['log10 ('+ FactorasString + ')'])
    y = np.array(newdf['log10(Niche Width)'])
    X = x[:, np.newaxis]
    degree = 1
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = sm.OLS(y, X_poly).fit()
    print(model.summary())
    # Generate smoother line
    x_smooth = np.linspace(x.min(), x.max(), 100)  # Generate more points along x-axis
    X_smooth = x_smooth[:, np.newaxis]
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth_pred = model.predict(X_smooth_poly)
    tolabel = {386585.0: 'E. coli',99287: 'S. enterica~Typhimurium',208964: 'P. aeruginosa',1313.0: 'S. pneumoniae'}
    
    scatter_plot = sns.lmplot(newdf, x='log10 ('+ FactorasString + ')', y='log10(Niche Width)',fit_reg = False, order = 1, scatter_kws={"s": 20, 'color': col }, line_kws = {'color': "000000", 'linewidth': 0.7})
    
    scatter_plot.set_xlabels('log10 ('+ FactorasString + ')', fontdict = {'size': 18})
    scatter_plot.set_ylabels('log10 (Niche Width)', fontdict = {'size': 18})
    plt.title('Effect of ' + FactorasString + '\n on niche width', fontdict = {'size': 18})
    plt.plot(x_smooth, y_smooth_pred, color = "red", linewidth = 2, label = 'Curve of best fit')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    a=2
    X2 = []
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
                    X2.append(x)
                    Y.append(y)
                    plt.text(x, y, tolabel[i], fontsize = 12, verticalalignment = va )
                    a += 1 
                else:
                     continue
    plt.scatter(X2,Y, c = '#000000', s = 5)
    plt.grid(True)
    plt.show()


Drawer('ProteinCodingGenes')