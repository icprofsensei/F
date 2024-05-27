import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE, trustworthiness
import matplotlib.pyplot as plt
import seaborn as sns
import random
class tSNElineage:
       def __init__(self, source, factor, rank):
                  self.source = source
                  self.factor = factor
                  self.rank = rank
       def prep(self):
            prok = pd.read_csv(self.source)
            nottouse = ['sci_name','cultures','sci_name', 'chromosome_numbers','genome_size','genome_gc_content','gene_counts','protein_coding_genes', 'lineage', '0', 'complex_medium', 'MediaID']
            nottouse.remove(self.factor)
            for col in prok.columns:
                  if col in nottouse:
                        del prok[col]
                  elif col.startswith('Unnamed'):
                        del prok[col]
                  else:
                        continue
            
            colnames = prok.columns.values
            names = []
            for index, row in prok.iterrows():
                              data = []
                              list  = row['lineage'].split()
                              names.append(list[self.rank])
            mydict = {}
            namesset = set(names)
            names2 = [i for i in namesset]
            for index, i in enumerate(names2):
                  mydict[i] = index
            print(mydict)
            overall = []
            for index, row in prok.iterrows():
                  data = []
                  for currentitem in row:
                        if currentitem == row['lineage']:
                                    val = mydict[currentitem.split()[self.rank]]
                                    data.append(val)
                        else:
                                    data.append(currentitem)
                  overall.append(data)
            df = pd.DataFrame(overall, columns = colnames)
            total = []
            for index, row in df.iterrows():
                  data = []
                  for item in row:
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
                  total.append(data)
            df2 = pd.DataFrame(total, columns =colnames )
            df2 = df2.fillna(0)
            df2.to_csv('Objective4tSNE/tSNE.csv')

            prok2 = pd.read_csv("Objective4tSNE/tSNE.csv")
            prok2.drop(prok2.columns[[0, 1]], axis=1, inplace=True)
            counts = prok2[self.factor].value_counts().to_frame()
            print(counts.head(20))
            count = 0
            #print(counts)
            for i in range(len(counts)):
                   #print(counts.iloc[i,0])
                   if counts.iloc[i,0] > 50:
                          count += 1
            print('Clusters', count)
            # Compute the dissimilarity matrix in a memory-efficient way
            sparse_data = csr_matrix(prok2)

            # Perform MiniBatch K-means clustering
            # n_clusters = n centroids, random_state = Seed for random number generation for initial centroid, batch_size = Size of mini batches used to process data. 
            kmeans = MiniBatchKMeans(n_clusters= count, random_state=42, batch_size=50)
            kmeans.fit(sparse_data)

            # Predict cluster labels
            #labels = kmeans.labels_
            labels = prok2[self.factor]
            label_mapping = {v: k for k, v in mydict.items()}
            #label_mapping = {0: ("Other"), 1: (str(self.factor) + " " + str(self.searchtaxa[0])), 2: (str(self.factor) + " " + str(self.searchtaxa[1])), 3: (str(self.factor) + " " + str(self.searchtaxa[2])), 4: (str(self.factor) + " " + str(self.searchtaxa[3]))}

            # Map the labels using the mapping
            mapped_labels = pd.Series(labels).map(label_mapping)

            # Perform t-SNE for dimensionality reduction
            # n_components = how many dimensions should the data be reduced to, random_state = Seed for random number generation, method = method used is barnes hut to accelerate processing of large datasets, n_iter = Max number of iterations for optimization, angle = (theta) = Trade-off parameter in barnes-hut. 0.5 is a balance between speed and accuracy. Lower means more accurate and low, higher means less accurate and quick. 
            #Decrease perplexity to increase cluster tightness. 
            #Increase early_exaggeration to increase separation between clusters initially. 
            tsne = TSNE(n_components=2, random_state=42, method='barnes_hut',perplexity = 20,learning_rate = 1000, n_iter=500, early_exaggeration=20, angle=0.5)
            tsne_coords = tsne.fit_transform(sparse_data.toarray())
            tw = trustworthiness(sparse_data.toarray(), tsne_coords)
            print(f"Trustworthiness of the t-SNE embedding: {tw:.4f}")

            # Create a DataFrame for easier plotting
            tsne_df = pd.DataFrame(tsne_coords, columns=['Dim1', 'Dim2'])
            tsne_df['Cluster'] = mapped_labels

            # Plot the t-SNE results with cluster labels
            colors = set()
            while len(colors) < len(names2):
                   color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                   colors.add(color)
            colors2 = [i for i in colors]
            plt.figure(figsize=(10, 8))
            scatter_plot = sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=tsne_df, palette=colors2, s=80)
            # Modify the legend title
            '''legend = scatter_plot.legend_
            legend.set_title(self.factor)'''


            plt.title('t-SNE Visualization of '+ self.factor)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            if self.rank == 0:
                   level = 'Phylum'
            elif self.rank ==1:
                   level = 'Class'
            elif self.rank ==2:
                   level = 'Order'
            else:
                   level = 'Genus'
            #plt.legend(bbox_to_anchor=(1.02, 1), loc = 'upper left', borderaxespad = 0.5, title= level)
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()