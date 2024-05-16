import pandas as pd
df = pd.read_csv('clean.csv')
df.join
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