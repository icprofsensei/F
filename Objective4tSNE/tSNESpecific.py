import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE, trustworthiness
import matplotlib.pyplot as plt
import seaborn as sns
import random
class tSNElineage:
       def __init__(self, source, factor, rank, searchtaxa):
                  self.source = source
                  self.factor = factor
                  self.rank = rank
                  self.searchtaxa = searchtaxa
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
                    if item == row[self.factor]:
                        if str(item).split()[self.rank] == self.searchtaxa[0]:
                              data.append(1)
                        elif str(item).split()[self.rank] == self.searchtaxa[1]:
                              data.append(2)
                        elif str(item).split()[self.rank] == self.searchtaxa[2]:
                              data.append(3)
                        elif str(item).split()[self.rank] == self.searchtaxa[3]:
                              data.append(4)
                        elif str(item).split()[self.rank] == self.searchtaxa[4]:
                              data.append(5)
                        elif str(item).split()[self.rank] == self.searchtaxa[5]:
                              data.append(6)
                        else:
                              data.append(0)


                    else:
                        if str(item).endswith('g/l'):
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
            kmeans = MiniBatchKMeans(n_clusters= len(self.searchtaxa) + 1, random_state=42, batch_size=50)
            kmeans.fit(sparse_data)

            # Predict cluster labels
            #labels = kmeans.labels_
            labels = prok2[self.factor]
            label_mapping = {0: ("Other"), 1: (str(self.factor) + " " + str(self.searchtaxa[0])), 2: (str(self.factor) + " " + str(self.searchtaxa[1])), 3: (str(self.factor) + " " + str(self.searchtaxa[2])), 4: (str(self.factor) + " " + str(self.searchtaxa[3])), 5: (str(self.factor) + " " + str(self.searchtaxa[4])), 6: (str(self.factor) + " " + str(self.searchtaxa[5])), }

            # Map the labels using the mapping
            mapped_labels = pd.Series(labels).map(label_mapping)

            # Perform t-SNE for dimensionality reduction
            # n_components = how many dimensions should the data be reduced to, random_state = Seed for random number generation, method = method used is barnes hut to accelerate processing of large datasets, n_iter = Max number of iterations for optimization, angle = (theta) = Trade-off parameter in barnes-hut. 0.5 is a balance between speed and accuracy. Lower means more accurate and low, higher means less accurate and quick. 
            tsne = TSNE(n_components=2, random_state=42, method='barnes_hut',perplexity = 60,learning_rate = 1000, n_iter=1500, early_exaggeration=20, angle=0.5)
            tsne_coords = tsne.fit_transform(sparse_data.toarray())
            tw = trustworthiness(sparse_data.toarray(), tsne_coords)
            print(f"Trustworthiness of the t-SNE embedding: {tw:.4f}")

            # Create a DataFrame for easier plotting
            tsne_df = pd.DataFrame(tsne_coords, columns=['Dim1', 'Dim2'])
            tsne_df['Cluster'] = mapped_labels


            plt.figure(figsize=(10, 8))
            scatter_plot = sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', edgecolor = 'none', data=tsne_df, palette=sns.color_palette("Set1",len(self.searchtaxa) + 1 ), s=50, legend='full')
            # Modify the legend title
            legend = scatter_plot.legend_
            legend.set_title(self.factor)


            plt.title('t-SNE Visualization of \n'+ self.factor, fontdict = {'size': 18})
            plt.xlabel('Dimension 1', fontdict = {'size': 18})
            plt.ylabel('Dimension 2', fontdict = {'size': 18})
            if self.rank == 0:
                   level = 'Phylum'
            elif self.rank ==1:
                   level = 'Class'
            elif self.rank ==2:
                   level = 'Order'
            else:
                   level = 'Genus'
            plt.legend(bbox_to_anchor=(1.02, 1), loc = 'upper left', borderaxespad = 0.5, title= level)
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
# Class
class tSNElineage2:
       def __init__(self, source, factor, rank, searchtaxa):
                  self.source = source
                  self.factor = factor
                  self.rank = rank
                  self.searchtaxa = searchtaxa
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
                    if item == row[self.factor]:
                        if str(item).split()[self.rank] == self.searchtaxa[0]:
                              data.append(1)
                        elif str(item).split()[self.rank] == self.searchtaxa[1]:
                              data.append(2)
                        elif str(item).split()[self.rank] == self.searchtaxa[2]:
                              data.append(3)
                        elif str(item).split()[self.rank] == self.searchtaxa[3]:
                              data.append(4)
                        elif str(item).split()[self.rank] == self.searchtaxa[4]:
                              data.append(5)
                        elif str(item).split()[self.rank] == self.searchtaxa[5]:
                              data.append(6)
                        elif str(item).split()[self.rank] == self.searchtaxa[6]:
                              data.append(7)
                        elif str(item).split()[self.rank] == self.searchtaxa[7]:
                              data.append(8)
                        
                        else:
                              data.append(0)


                    else:
                        if str(item).endswith('g/l'):
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
            kmeans = MiniBatchKMeans(n_clusters= len(self.searchtaxa) + 1, random_state=42, batch_size=50)
            kmeans.fit(sparse_data)

            # Predict cluster labels
            #labels = kmeans.labels_
            labels = prok2[self.factor]
            label_mapping = {0: ("Other"), 1: (str(self.factor) + " " + str(self.searchtaxa[0])), 2: (str(self.factor) + " " + str(self.searchtaxa[1])), 3: (str(self.factor) + " " + str(self.searchtaxa[2])), 4: (str(self.factor) + " " + str(self.searchtaxa[3])), 5: (str(self.factor) + " " + str(self.searchtaxa[4])), 6: (str(self.factor) + " " + str(self.searchtaxa[5])), 7: (str(self.factor) + " " + str(self.searchtaxa[6])), 8: (str(self.factor) + " " + str(self.searchtaxa[7]))}

            # Map the labels using the mapping
            mapped_labels = pd.Series(labels).map(label_mapping)

            # Perform t-SNE for dimensionality reduction
            # n_components = how many dimensions should the data be reduced to, random_state = Seed for random number generation, method = method used is barnes hut to accelerate processing of large datasets, n_iter = Max number of iterations for optimization, angle = (theta) = Trade-off parameter in barnes-hut. 0.5 is a balance between speed and accuracy. Lower means more accurate and low, higher means less accurate and quick. 
            tsne = TSNE(n_components=2, random_state=42, method='barnes_hut',perplexity = 60,learning_rate = 1000, n_iter=2000, early_exaggeration=20, angle=0.5)
            tsne_coords = tsne.fit_transform(sparse_data.toarray())
            tw = trustworthiness(sparse_data.toarray(), tsne_coords)
            print(f"Trustworthiness of the t-SNE embedding: {tw:.4f}")

            # Create a DataFrame for easier plotting
            tsne_df = pd.DataFrame(tsne_coords, columns=['Dim1', 'Dim2'])
            tsne_df['Cluster'] = mapped_labels


            plt.figure(figsize=(10, 8))
            scatter_plot = sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=tsne_df, palette=sns.color_palette("Paired",len(self.searchtaxa) + 1 ), s=50, legend='full')
            # Modify the legend title
            legend = scatter_plot.legend_
            legend.set_title(self.factor)


            plt.title('t-SNE Visualization of \n'+ self.factor, fontdict = {'size': 18})
            plt.xlabel('Dimension 1', fontdict = {'size': 18})
            plt.ylabel('Dimension 2', fontdict = {'size': 18})
            if self.rank == 0:
                   level = 'Phylum'
            elif self.rank ==1:
                   level = 'Class'
            elif self.rank ==2:
                   level = 'Order'
            else:
                   level = 'Genus'
            plt.legend(bbox_to_anchor=(1.02, 1), loc = 'upper left', borderaxespad = 0.5, title= level)
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
# Order
class tSNElineage3:
       def __init__(self, source, factor, rank, searchtaxa):
                  self.source = source
                  self.factor = factor
                  self.rank = rank
                  self.searchtaxa = searchtaxa
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
                    if item == row[self.factor]:
                        if str(item).split()[self.rank] == self.searchtaxa[0]:
                              data.append(1)
                        elif str(item).split()[self.rank] == self.searchtaxa[1]:
                              data.append(2)
                        elif str(item).split()[self.rank] == self.searchtaxa[2]:
                              data.append(3)
                        elif str(item).split()[self.rank] == self.searchtaxa[3]:
                              data.append(4)
                        elif str(item).split()[self.rank] == self.searchtaxa[4]:
                              data.append(5)
                        elif str(item).split()[self.rank] == self.searchtaxa[5]:
                              data.append(6)
                        elif str(item).split()[self.rank] == self.searchtaxa[6]:
                              data.append(7)
                        elif str(item).split()[self.rank] == self.searchtaxa[7]:
                              data.append(8)
                        elif str(item).split()[self.rank] == self.searchtaxa[8]:
                              data.append(9)
                        elif str(item).split()[self.rank] == self.searchtaxa[9]:
                              data.append(10)
                        elif str(item).split()[self.rank] == self.searchtaxa[10]:
                              data.append(11)
                        elif str(item).split()[self.rank] == self.searchtaxa[11]:
                              data.append(12)
                        elif str(item).split()[self.rank] == self.searchtaxa[12]:
                              data.append(13)
                        elif str(item).split()[self.rank] == self.searchtaxa[13]:
                              data.append(14)
                        elif str(item).split()[self.rank] == self.searchtaxa[14]:
                              data.append(15)
                        elif str(item).split()[self.rank] == self.searchtaxa[15]:
                              data.append(16)
                        elif str(item).split()[self.rank] == self.searchtaxa[16]:
                              data.append(17)
                        elif str(item).split()[self.rank] == self.searchtaxa[17]:
                              data.append(18)
                        elif str(item).split()[self.rank] == self.searchtaxa[18]:
                              data.append(19)
                        else:
                              data.append(0)


                    else:
                        if str(item).endswith('g/l'):
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
            kmeans = MiniBatchKMeans(n_clusters= len(self.searchtaxa) + 1, random_state=42, batch_size=50)
            kmeans.fit(sparse_data)

            # Predict cluster labels
            #labels = kmeans.labels_
            labels = prok2[self.factor]
            label_mapping = {0: ("Other"), 1: (str(self.factor) + " " + str(self.searchtaxa[0])), 2: (str(self.factor) + " " + str(self.searchtaxa[1])), 3: (str(self.factor) + " " + str(self.searchtaxa[2])), 4: (str(self.factor) + " " + str(self.searchtaxa[3])), 5: (str(self.factor) + " " + str(self.searchtaxa[4])), 6: (str(self.factor) + " " + str(self.searchtaxa[5])), 7: (str(self.factor) + " " + str(self.searchtaxa[6])), 8: (str(self.factor) + " " + str(self.searchtaxa[7])), 9: (str(self.factor) + " " + str(self.searchtaxa[8])), 10: (str(self.factor) + " " + str(self.searchtaxa[9])), 11: (str(self.factor) + " " + str(self.searchtaxa[10])), 12: (str(self.factor) + " " + str(self.searchtaxa[11])), 13: (str(self.factor) + " " + str(self.searchtaxa[12])), 14: (str(self.factor) + " " + str(self.searchtaxa[13])), 15: (str(self.factor) + " " + str(self.searchtaxa[14])), 16: (str(self.factor) + " " + str(self.searchtaxa[15])), 17: (str(self.factor) + " " + str(self.searchtaxa[16])), 18: (str(self.factor) + " " + str(self.searchtaxa[17])), 19: (str(self.factor) + " " + str(self.searchtaxa[18]))}

            # Map the labels using the mapping
            mapped_labels = pd.Series(labels).map(label_mapping)

            # Perform t-SNE for dimensionality reduction
            # n_components = how many dimensions should the data be reduced to, random_state = Seed for random number generation, method = method used is barnes hut to accelerate processing of large datasets, n_iter = Max number of iterations for optimization, angle = (theta) = Trade-off parameter in barnes-hut. 0.5 is a balance between speed and accuracy. Lower means more accurate and low, higher means less accurate and quick. 
            tsne = TSNE(n_components=2, random_state=42, method='barnes_hut',perplexity = 60,learning_rate = 1000, n_iter=2000, early_exaggeration=20, angle=0.5)
            tsne_coords = tsne.fit_transform(sparse_data.toarray())
            tw = trustworthiness(sparse_data.toarray(), tsne_coords)
            print(f"Trustworthiness of the t-SNE embedding: {tw:.4f}")

            # Create a DataFrame for easier plotting
            tsne_df = pd.DataFrame(tsne_coords, columns=['Dim1', 'Dim2'])
            tsne_df['Cluster'] = mapped_labels


            plt.figure(figsize=(10, 8))
            scatter_plot = sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=tsne_df, palette=sns.color_palette("tab20",len(self.searchtaxa) + 1 ), s=50, legend='full')
            # Modify the legend title
            legend = scatter_plot.legend_
            legend.set_title(self.factor)


            plt.title('t-SNE Visualization of \n'+ self.factor, fontdict = {'size': 18})
            plt.xlabel('Dimension 1', fontdict = {'size': 18})
            plt.ylabel('Dimension 2', fontdict = {'size': 18})
            if self.rank == 0:
                   level = 'Phylum'
            elif self.rank ==1:
                   level = 'Class'
            elif self.rank ==2:
                   level = 'Order'
            else:
                   level = 'Genus'
            plt.legend(bbox_to_anchor=(1.02, 1), loc = 'upper left', borderaxespad = 0.5, title= level)
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
'''
                        elif str(item).split()[self.rank] == self.searchtaxa[6]:
                              data.append(7)
                        elif str(item).split()[self.rank] == self.searchtaxa[7]:
                              data.append(8)
'''
#7: (str(self.factor) + " " + str(self.searchtaxa[6])), 8: (str(self.factor) + " " + str(self.searchtaxa[7]))