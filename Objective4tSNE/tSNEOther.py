import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE, trustworthiness
import matplotlib.pyplot as plt
import seaborn as sns
class tSNElineage:
       def __init__(self, source, factor, rank, searchtaxa, groups):
                  self.source = source
                  self.factor = factor
                  self.rank = rank
                  self.searchtaxa = searchtaxa
                  self.groups = groups
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
            '''for index, row in prok.iterrows():
                  print(row['lineage'])'''
            for index, row in prok.iterrows():
                  data = []
                  for item in row:
                        
                        if str(item).split()[self.rank] == self.searchtaxa[0]:
                              data.append(1)
                        elif str(item).split()[self.rank] == self.searchtaxa[1]:
                              data.append(2)
                        elif str(item).split()[self.rank] == self.searchtaxa[2]:
                              data.append(3)
                        elif str(item).split()[self.rank] == self.searchtaxa[3]:
                              data.append(4)
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
            df.to_csv('tSNE.csv')

            prok2 = pd.read_csv("tSNE.csv")
            prok2.drop(prok2.columns[[0, 1]], axis=1, inplace=True)
            
            # Compute the dissimilarity matrix in a memory-efficient way
            sparse_data = csr_matrix(prok2)

            # Perform MiniBatch K-means clustering
            # n_clusters = n centroids, random_state = Seed for random number generation for initial centroid, batch_size = Size of mini batches used to process data. 
            kmeans = MiniBatchKMeans(n_clusters= self.groups, random_state=42, batch_size=50)
            kmeans.fit(sparse_data)

            # Predict cluster labels
            #labels = kmeans.labels_
            labels = prok2[self.factor]
            label_mapping = {0: ("Other"), 1: (str(self.factor) + " " + str(self.searchtaxa[0])), 2: (str(self.factor) + " " + str(self.searchtaxa[1])), 3: (str(self.factor) + " " + str(self.searchtaxa[2])), 4: (str(self.factor) + " " + str(self.searchtaxa[3]))}

            # Map the labels using the mapping
            mapped_labels = pd.Series(labels).map(label_mapping)

            # Perform t-SNE for dimensionality reduction
            # n_components = how many dimensions should the data be reduced to, random_state = Seed for random number generation, method = method used is barnes hut to accelerate processing of large datasets, n_iter = Max number of iterations for optimization, angle = (theta) = Trade-off parameter in barnes-hut. 0.5 is a balance between speed and accuracy. Lower means more accurate and low, higher means less accurate and quick. 
            tsne = TSNE(n_components=2, random_state=42, method='barnes_hut', n_iter=300, angle=0.5)
            tsne_coords = tsne.fit_transform(sparse_data.toarray())
            tw = trustworthiness(sparse_data.toarray(), tsne_coords)
            print(f"Trustworthiness of the t-SNE embedding: {tw:.4f}")

            # Create a DataFrame for easier plotting
            tsne_df = pd.DataFrame(tsne_coords, columns=['Dim1', 'Dim2'])
            tsne_df['Cluster'] = mapped_labels

            # Plot the t-SNE results with cluster labels
            plt.figure(figsize=(10, 8))
            scatter_plot = sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=tsne_df, palette='Set1', s=100, legend='full')

            # Modify the legend title
            legend = scatter_plot.legend_
            legend.set_title(self.factor)


            plt.title('t-SNE Visualization of '+ self.factor)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend(title='Cluster')
            plt.grid(True)
            plt.show()