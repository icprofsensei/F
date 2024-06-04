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
    newdf['bins'] = pd.qcut(newdf['log10 ('+ FactorasString + ')'], q = 12)
    #tolabel = {386585.0: 'E. coli',99287: 'S. enterica~Typhimurium',208964: 'P. aeruginosa',1313.0: 'S. pneumoniae'}

    # Group by bins and calculate the range of cat1 values in each bin
    bin_ranges = newdf.groupby('bins')['log10(Niche Width)'].apply(lambda x: max(x) - min(x))

    # Convert bin intervals to string labels
    bin_labels = [str(bin_range).split(',')[0].strip('(')[0:5] for bin_range in bin_ranges.index]
    print(bin_labels)
    # Plot the range of cat1 values for each bin of cat2 as a bar chart
    plt.bar(bin_labels, bin_ranges, width=1, color = col)

    # Set the labels and title
    plt.xlabel('log10 ('+ FactorasString + ')', fontsize = 18, labelpad= 10)
    plt.ylabel('Range of log10(Niche Width)', fontsize = 18)
    plt.title('Range of log10(Niche Width) within \n '+ 'log10 ('+ FactorasString + ')' + ' bins', fontsize = 18)
    plt.xlim(0, len(bin_ranges) - 1)
    plt.xticks(range(len(bin_ranges)), bin_labels,rotation=45)
    plt.tight_layout()
    plt.show()
Drawer('GeneCounts')