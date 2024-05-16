#PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats

# Gold = '#ffd700'
class PCAProk:
    def __init__(self, factor, cutoff, zthresh1, zthresh2):
                self.factor = factor
                self.cutoff = cutoff
                self.zthresh1 = zthresh1
                self.zthresh2 = zthresh2
    def figmaker(self):
                        
                prok = pd.read_csv("PCA.csv")
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
                label1 = str(self.factor) + '>'+ str(self.cutoff)
                label2 = str(self.factor) + '<='+ str(self.cutoff)
                plt.scatter(no_outliers['principal component 1'], no_outliers['principal component 2'], s = 2,c= prok[self.factor], alpha = 0.5, label = prok[self.factor])
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.title('Principal Component Analysis of Prokaryote ingredient requirements dataset')
                plt.legend([label1, label2])
                plt.show()