import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('base.csv')
#W/o rescaling
#examine covariance btw random pairs of columns + produce histogram
# Are distributions skewed (species A vs Species B)
# Repeat with log transformed datasets (Convert 0 --> 1s)
# What are good datasets for a PCA. 
# Alt = Try NMDS 
# How do other people find ways to classify data with lots of 0. 
# w/ rescaling repeat. 
# Niche conservatism = Niches are conserved phylogenetically. 
# Are more similar strains clustered together?
# What are the distributions of niche widths? Generalists , specialists. In a histogram. 
# --> Repeat for subclassifications such as carbon sources, n sources. 
# Relationship between genome size, gene count, protein coding genes (fundamental niche) and realised niche
 # Niche conservatism 
# Aditi Madkaikar (Slack)

'''
Objective 1: Demonstrate niche distributions based on resources.
Objective 2: How are niche distributions correlated with fundamental niche?
Objective 3: Niche conservatism
'''
df.drop(['Unnamed: 0','taxid', 'genome_size','gene_counts','protein_coding_genes','min_pH','max_pH', 'Distilled water'], axis = 1, inplace = True)
print(list(df.iloc[10]))
#plt.scatter(np.log10(df['GenomeSize']), np.log10(df['NicheWidth']), c = '#0F52BA', s = 10, alpha = 0.5)
plt.scatter(list(df.iloc[22]), list(df.iloc[180]), c = '#0F52BA', s = 10, alpha = 0.5)
plt.title('Effect of genome size on niche width')
plt.xlabel('row22')
plt.ylabel('row180')
plt.show()

#Rewrite hypotheses
#Shorten intro
# 2nd hyp = Niche conservatism
# Have 3 objectives and 2 hypotheses (1. Niche distribution, 2 . Niche conservatism)
# Give a draft by next Monday

# For the resource niche widths for objective 1--> Plot the log distribution. (Is it more gaussian?)
# Make sure axis labels are large (same size as the text)
# De Long et al paper on genome size and growth rate correlation --> Basis of hypothesis
# Chris Kempis paper (size scaling laws papers)
# Could try to fit a polynomial regression (is there a peak in the middle) --> Shows statistical certainty to all three relationships
# Show certain species on the axes. 

# Potentially move the significant ingredients to supplementary ingredients. 