import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE, trustworthiness
import matplotlib.pyplot as plt
import seaborn as sns
import random

import pandas as pd

def drawer (factor, rank, perp, ee, lr):
                prok = pd.read_csv('clean.csv')
                prok2 = pd.read_csv("Objective4tSNE/tSNE.csv")
                prok2.drop(prok2.columns[[0, 1]], axis=1, inplace=True)
                names = []
                for index, row in prok.iterrows():
                              list  = row['lineage'].split()
                              names.append(list[rank])
                mydict = {}
                namesset = set(names)
                names2 = [i for i in namesset]
                for index, i in enumerate(names2):
                    mydict[i] = index
                    
                counts = prok2[factor].value_counts().to_frame()
                count = 0
                for i in range(len(counts)):
                        if counts.iloc[i,0] > 50:
                                count += 1
                print('Clusters', count)
                sparse_data = csr_matrix(prok2)
                kmeans = MiniBatchKMeans(n_clusters= count, random_state=42, batch_size=50)
                kmeans.fit(sparse_data)
                labels = prok2[factor]
                label_mapping = {v: k for k, v in mydict.items()}
                mapped_labels = pd.Series(labels).map(label_mapping)        
                tsne = TSNE(n_components=2, random_state=42, method='barnes_hut',perplexity = perp,learning_rate = lr, n_iter=300, early_exaggeration=ee, angle=0.9)
                tsne_coords = tsne.fit_transform(sparse_data.toarray())
                tw = trustworthiness(sparse_data.toarray(), tsne_coords)
                print(f"Trustworthiness of the t-SNE embedding: {tw:.4f}")
                tsne_df = pd.DataFrame(tsne_coords, columns=['Dim1', 'Dim2'])
                tsne_df['Cluster'] = mapped_labels
                colors = set()
                while len(colors) < len(names2):
                        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                        colors.add(color)
                colors2 = [i for i in colors]
                plt.figure(figsize=(10, 8))
                scatter_plot = sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=tsne_df, palette=colors2, s=10, legend='full')
                legend = scatter_plot.legend_
                legend.set_title(factor)
                plt.title('t-SNE Visualization of '+ factor)
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                if rank == 0:
                        level = 'Phylum'
                elif rank ==1:
                        level = 'Class'
                elif rank ==2:
                        level = 'Order'
                else:
                        level = 'Genus'
                plt.legend(bbox_to_anchor=(1.02, 1), loc = 'upper left', borderaxespad = 1, title= level)
                plt.grid(True)
                plt.show()
drawer('lineage', 0, 60, 24, 900)
# perp = 60, early exag = 24, lr = 900