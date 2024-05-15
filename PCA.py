#PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import csv
with open('clean.csv', 'r') as file:
    csv_reader = csv.reader(file)
    ncol = len(next(csv_reader))


prok = pd.read_csv("prok_ing.csv")
row_count = prok.shape[0]

for ind in prok.index:
    pcg = (prok['protein_coding_genes'][ind])
    if pcg >= 2000:
        prok['protein_coding_genes'][ind] = 1
    else:
        prok['protein_coding_genes'][ind] = 0
print("rows:" + str(row_count) + " cols: " + str(ncol))
features = list(prok.columns.values)
cultures = len(features) - 8
tax_id = len(features) - 7
sci_name = len(features) - 6
chromosome_numbers = len(features) -5
genome_size = len(features) - 4
genome_gc_content = len(features) -3
gene_counts = len(features) -2
protein_coding_genes = len(features) -1 
prok.drop(prok.columns[[0, 1, 2, cultures, tax_id, sci_name, chromosome_numbers, genome_size, genome_gc_content, gene_counts]], axis=1, inplace=True)
#print(prok.head)
#print(prok.shape)
features = list(prok.columns.values)
x = prok.loc[:, features ].values
x = StandardScaler().fit_transform(x)
print(x.shape)
print(np.mean(x), np.std(x))
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_prok = pd.DataFrame(x, columns = feat_cols)
print(prok.tail())
pca_prok = PCA(2)
principalComponents_Prok = pca_prok.fit_transform(x)
principal_prok_df = pd.DataFrame(data = principalComponents_Prok, columns = ['principal component 1', 'principal component 2'])
#Find outliers using Z score
z = np.abs(stats.zscore(principal_prok_df['principal component 1']))
threshold_z = 10
outlier_indices = np.where(z > threshold_z)[0]
print('Outlier indices:', outlier_indices)
no_outliers = principal_prok_df.drop(outlier_indices)
prok = prok.drop(outlier_indices)
#print(principal_prok_df.tail())
print('Explained variation per principal component: {}'.format(pca_prok.explained_variance_ratio_))
plt.scatter(no_outliers['principal component 1'], no_outliers['principal component 2'], s = 2,c= prok['protein_coding_genes'], alpha = 0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis of Prokaryote ingredient requirements dataset')
plt.legend(prok['protein_coding_genes'])
plt.show()