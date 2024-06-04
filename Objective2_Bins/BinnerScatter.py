#Construct basic intrinsic values plots against number of media which a species can be cultured on
import pandas as pd
df = pd.read_csv("Objective2/NWidth.csv")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
def drop_lower(df_group):
    df_group['rank'] = df_group['log10(Niche Width)'].rank()
    df_group = df_group[df_group['rank'] > df_group['rank'].quantile(0.90)]  # Drop lower 90% of data in each bin
    return df_group
    
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
    newdf['bins'] = pd.qcut(newdf['log10 ('+ FactorasString + ')'], q = 12)
    #tolabel = {386585.0: 'E. coli',99287: 'S. enterica~Typhimurium',208964: 'P. aeruginosa',1313.0: 'S. pneumoniae'}
    #groupby = Reshapes the dataframe with new rows equating to the grouping factor (in this case it is bins of the factor) containing data of both the grouping factor and the niche width.
    #apply  = Inverts the rows to columns and then applies the function (specified in the parameter to each item in the column)
    #the drop lower function forms a new group containing the data ranked by niche width. It only allows data in the top 10% to be included in the group. 
    df_filtered = newdf.groupby('bins').apply(drop_lower)
    # Modifies the datafrmae in place (no new df) and resets all index.
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.drop('bins', axis=1, inplace=True) # Axis = 1 (drop a column - bins). 
    newdf2 = pd.DataFrame(zip(df_filtered['log10 ('+ FactorasString + ')'],df_filtered['log10(Niche Width)']), columns = ['head1', 'head2'])
    x = np.array(newdf2['head1'])
    y = np.array(newdf2['head2'])
    X = x[:, np.newaxis]
    degree = 2
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = sm.OLS(y, X_poly).fit()
    y_pred = model.predict(X_poly)
    print(model.summary())

    # Generate smoother line
    x_smooth = np.linspace(x.min(), x.max(), 100)  # Generate more points along x-axis
    X_smooth = x_smooth[:, np.newaxis]
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth_pred = model.predict(X_smooth_poly)


    scatter_plot = sns.lmplot(df_filtered, x='log10 ('+ FactorasString + ')', y='log10(Niche Width)',fit_reg = False, scatter_kws={"s": 20, 'color': col })
    plt.plot(x_smooth, y_smooth_pred, color = "red", linewidth = 2, label = 'Curve of best fit')
    plt.xlabel('log10 ('+ FactorasString + ')', fontdict = {'size': 18})
    plt.ylabel('log10 (Niche Width)', fontdict = {'size': 18})
    plt.title('Effect of ' + FactorasString + ' on niche width \n ~ upper 10% quantile plot', fontdict = {'size': 18})
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.grid(True)
    plt.show()
Drawer('GenomeSize')