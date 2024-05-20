#PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
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
                prok2 = prok
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
                prok = prok.drop(outliers)
                print('Explained variation per principal component: {}'.format(pca_prok.explained_variance_ratio_))
                #labels = kmeans.labels_
                labels = prok2[self.factor]
                label_mapping = {0: (str(self.factor) + "<=" + str(self.cutoff)), 1: (str(self.factor) + ">" + str(self.cutoff))}
                # Map the labels using the mapping
                mapped_labels = pd.Series(labels).map(label_mapping)
                # Add to the no_outliers df. 
                no_outliers['Cluster'] = mapped_labels
                #Draw Plot
                plt.figure(figsize=(10, 8))
                scatter_plot = sns.scatterplot(x='principal component 1', y='principal component 2', hue='Cluster', data=no_outliers, palette='Set1', s=100, legend='full')
                #plt.scatter(no_outliers['principal component 1'], no_outliers['principal component 2'], s = 2,c= prok[self.factor], alpha = 0.5, label = prok[self.factor])

                # Modify the legend title
                legend = scatter_plot.legend_
                legend.set_title(self.factor)

                plt.title('PCA Visualization of '+ self.factor)
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.legend(title='Cluster')
                plt.grid(True)
                plt.show()