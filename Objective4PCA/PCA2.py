#PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import seaborn as sns
from scipy import stats
# Gold = '#ffd700'
class PCAProk:
    def __init__(self, factor, cutoff, zthresh1, zthresh2):
                self.factor = factor
                self.cutoff = cutoff
                self.zthresh1 = zthresh1
                self.zthresh2 = zthresh2
    def figmaker(self):
                        
                prok = pd.read_csv("Objective4PCA/PCA.csv")
                for ind in prok.index:
                    pcg = (prok[self.factor][ind])
                    if pcg >= self.cutoff:
                        prok[self.factor][ind] = 1
                    else:
                        prok[self.factor][ind] = 0
                prok.drop(prok.columns[[0, 1]], axis=1, inplace=True)
                
                
                scaler = StandardScaler()
                prok2 = scaler.fit_transform(prok)
                sparse_matrix = csr_matrix(prok2)
                # Initialize TruncatedSVD
                n_components = 2
                svd = TruncatedSVD(n_components=n_components)

                # Fit and transform the data
                sparse_matrix_reduced = svd.fit_transform(sparse_matrix)
                print(f"Explained variance ratio: {svd.explained_variance_ratio_}")
                print(f"Shape of reduced matrix: {sparse_matrix_reduced.shape}")
                df_reduced = pd.DataFrame(sparse_matrix_reduced, columns=['Principal Component 1', 'Principal Component 2'])

                z1 = np.abs(stats.zscore(df_reduced['Principal Component 1']))
                outlier_indices1 = np.where(z1 > self.zthresh1)[0]
                z2 = np.abs(stats.zscore(df_reduced['Principal Component 2']))
                outlier_indices2 = np.where(z2 > self.zthresh2)[0]
                outliers1 = [o for o in outlier_indices1]
                outliers2 = [o for o in outlier_indices2]
                outliers = set(outliers1 + outliers2)
                outliers = list(outliers)
                print('Outlier indices:', outliers)
                no_outliers = df_reduced.drop(outliers)
                prok.drop(outliers)
                labels = prok[self.factor]
                label_mapping = {0: (str(self.factor) + "<=" + str(self.cutoff)), 1: (str(self.factor) + ">" + str(self.cutoff))}
                # Map the labels using the mapping
                mapped_labels = pd.Series(labels).map(label_mapping)
                # Add to the no_outliers df. 
                no_outliers['Cluster'] = mapped_labels
                #Draw Plot
                plt.figure(figsize=(10, 8))
                scatter_plot = sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Cluster', data=no_outliers, palette='Set1', s=100, legend='full')

                # Modify the legend title
                legend = scatter_plot.legend_
                legend.set_title(self.factor)

                plt.title('Log10 Scaled PCA Visualization of '+ self.factor)
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.legend(title='Cluster')
                plt.grid(True)
                plt.show()