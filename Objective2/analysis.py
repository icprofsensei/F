#Construct basic intrinsic values plots against number of media which a species can be cultured on
import pandas as pd
df = pd.read_csv("Objective2/NWidth.csv")
import matplotlib.pyplot as plt
import numpy as np

#Colours
#Emerald green  = Protein Coding Genes = #009B77
# Ruby = Gene Counts = #e0115f
# Saphire Blue = Genome Size = #0F52BA


#Drawing scatter plots (outliers included)

plt.scatter(np.log10(df['GenomeSize']), np.log10(df['NicheWidth']), c = '#0F52BA', s = 10, alpha = 0.5)
plt.title('Effect of genome size on niche width')
plt.xlabel('log10 (GenomeSize)')
plt.ylabel('log 10 (Niche Width)')
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

