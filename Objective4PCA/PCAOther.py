import pandas as pd
from PCA import PCAProk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
class PCAlineage:
       def __init__(self, source, factor, rank, searchtaxa, zthresh1, zthresh2):
                  self.source = source
                  self.factor = factor
                  self.rank = rank
                  self.searchtaxa = searchtaxa
                  self.zthresh1 = zthresh1
                  self.zthresh2 = zthresh2
       def prep(self):
            prok = pd.read_csv(self.source)
            nottouse = ['sci_name','cultures','sci_name','chromosome_numbers','genome_size','genome_gc_content', 'protein_coding_genes', 'gene_counts', 'lineage', '0', 'complex_medium', 'MediaID']
            nottouse.remove(self.factor)
            for col in prok.columns:
                  if col in nottouse:
                        del prok[col]
                  elif col.startswith('Unnamed'):
                        del prok[col]
                  else:
                        continue
            overall = []
            for index, row in prok.iterrows():
                  data = []
                  for item in row:
                        if str(item).split()[self.rank] == self.searchtaxa:
                              data.append(1)
                        elif str(item).endswith('g/l'):
                              data.append(float(str(item).rstrip('g/l')))
                        elif str(item).endswith('g'):
                              data.append(float(str(item).rstrip('g')))
                        elif str(item).endswith('ml'):
                              data.append(float(str(item).rstrip('ml')))
                        elif type(item) == float or type(item) == int:
                              data.append(item)
                        else:
                              data.append(0)
                  overall.append(data)
            df = pd.DataFrame(overall, columns =list(prok.columns.values) )
            df = df.fillna(0)
            df.to_csv('Objective4PCA/PCA.csv')
            PCADF = pd.read_csv("Objective4PCA/PCA.csv")
            prok2 = PCADF
            prok2.drop(prok2.columns[[0, 1]], axis=1, inplace=True)
            features = list(prok2.columns.values)
            x = prok2.loc[:, features ].values
            x = StandardScaler().fit_transform(x)
            print(x.shape)
            print(np.mean(x), np.std(x))
            pca_prok = PCA(2)
            principalComponents_Prok = pca_prok.fit_transform(x)
            principal_prok_df = pd.DataFrame(data = principalComponents_Prok, columns = ['principal component 1', 'principal component 2'])
            print('Explained variation per principal component: {}'.format(pca_prok.explained_variance_ratio_))
            #Find outliers using Z score
            z1 = np.abs(stats.zscore(principal_prok_df['principal component 1']))
            outlier_indices1 = np.where(z1 > self.zthresh1)[0]
            z2 = np.abs(stats.zscore(principal_prok_df['principal component 2']))
            outlier_indices2 = np.where(z2 > self.zthresh2)[0]
            outliers1 = [o for o in outlier_indices1]
            outliers2 = [o for o in outlier_indices2]
            outliers = set(outliers1 + outliers2)
            outliers = list(outliers)
            print('Outlier indices:', outliers)
            no_outliers = principal_prok_df.drop(outliers)
            PCADF = PCADF.drop(outliers)
            print('Explained variation per principal component: {}'.format(pca_prok.explained_variance_ratio_))
            label1 = str(self.factor) +" " + str(self.searchtaxa)
            label2 = str(self.factor) + 'other'
            #, labels = [label2, label1]
            scatter = plt.scatter(no_outliers['principal component 1'], no_outliers['principal component 2'], s = 2,c= PCADF[self.factor], alpha = 0.5)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('Principal Component Analysis of Prokaryote ingredient requirements dataset')
            plt.legend(*scatter.legend_elements())
            plt.show()