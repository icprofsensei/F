This repository contains data and python scripts for my paper on 'Quantifying the Prokaryotic Resource Niche'. 
The data is split into two main files: 
clean.csv is the main dataset which contains each species, its growth medium, fundamental niche characteristics and other growth media information.
base.csv contains same information as clean.csv, however units are removed for easier numerical comparisons, additional growth media are also removed. 

The repository is divided into my 4 main study objectives. Objective (1),(2) and (4) are discussed in the paper. 
1) Distribution of the prokaryotic resource niche
2) Correlation analysis with the prokaryotic resource niche and the fundamental niche. There is also a binned folder which unpacks the distributions further to find quantile regressions. 
3) Significant ingredient analysis.
4) Cluster analysis  - split over the PCA  and tSNE approaches.

Reference jsons contains the reference files supplied by the DSMZ. 
problemtaxa.txt contains all the species which were excluded from this study but were mentioned in the database. Their exclusion was based on data insufficiencies which would have reduced dataset completeness. 

