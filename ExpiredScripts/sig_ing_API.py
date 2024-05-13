
import requests
import pandas as pd
from scipy import stats
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
sig_genecount = pd.read_csv("sig_genecount.csv")
sig_genecount = sig_genecount.reset_index()
with open('sig_ingdf.csv', 'a+') as sg:
    w = csv.writer(sg)
    for index, row in sig_genecount.iterrows():
        id = str((row['id']))
        url = 'https://mediadive.dsmz.de/rest/ingredient/' + str(id)
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                item = response.json()
                data = (item['data'])
                if row['cor_type'] == 'Pos':
                    val = 1
                elif row['cor_type'] == 'Neg':
                    val = 0
                entry = {'name': row['name'], 'correlation_type': val, 'complex_compound': data['complex_compound'],'prevalence_in_media':len(data['media']) }
                w.writerow(entry.values())
            except:
                #print('Missing data entries for lineage')
                print(id, ' Missing data entries for ingredient')
                continue
        else:
            continue
# read contents of csv file 
file = pd.read_csv("sig_ingdf.csv") 

  
# adding header 
headerList = ['name', 'correlation_type', 'complex_compound', 'prevalence_in_media'] 
  
# converting data frame to csv 
file.to_csv("sig_ingdf.csv", header=headerList, index=False) 

ing = pd.read_csv('sig_ingdf.csv')
ing2 = pd.read_csv('sig_ingdf.csv')
ing.drop(ing.columns[[0, 1]], axis=1, inplace=True)
features = list(ing.columns.values)
x = ing.loc[:, features ].values
x = StandardScaler().fit_transform(x)
print(x.shape)
print(np.mean(x), np.std(x))
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_ing = pd.DataFrame(x, columns = feat_cols)
print(ing.tail())
pca_ing = PCA(2)
principalComponents_Prok = pca_ing.fit_transform(x)
principal_ing_df = pd.DataFrame(data = principalComponents_Prok, columns = ['principal component 1', 'principal component 2'])
#Find outliers using Z score
z = np.abs(stats.zscore(principal_ing_df['principal component 1']))
threshold_z = 10
outlier_indices = np.where(z > threshold_z)[0]
print('Outlier indices:', outlier_indices)
no_outliers = principal_ing_df.drop(outlier_indices)
prok = ing.drop(outlier_indices)
#print(principal_prok_df.tail())
print('Explained variation per principal component: {}'.format(pca_ing.explained_variance_ratio_))
plt.scatter(no_outliers['principal component 1'], no_outliers['principal component 2'], s = 2, c= ing2['correlation_type'], alpha = 0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis of significant ingredients in dataset')
plt.legend(ing2['correlation_type'])
plt.show()