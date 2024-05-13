#Construct basic intrinsic values plots against number of media which a species can be cultured on
import pandas as pd
from ast import literal_eval
df = pd.read_csv("prokdatacomplete.csv")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

#df.to_csv('prokdatacomplete.csv')
#print(df.head())
# Add extra column next to cultures to save the number of different acceptable media per species. 
number_of_media = []
for index, row in df.iterrows():
    cultures = literal_eval(row['cultures'])
    length = len(cultures)
    number_of_media.append(length)
df.insert(3, 'Number_of_media', number_of_media, True)
#Colours
#Emerald green  = Protein Coding Genes = #009B77
# Ruby = Gene Counts = #e0115f
# Saphire Blue = Genome Size = #0F52BA


#Drawing scatter plots (outliers included)

plt.scatter(np.log10(df['protein_coding_genes']), np.log10(df['Number_of_media']), c = '#009B77', s = 10, alpha = 0.5)
plt.title('Effect of protein coding genes on niche width')
plt.xlabel('log10 (Protein Coding Genes)')
plt.ylabel('log 10 (Number of different media)')
plt.show()


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

