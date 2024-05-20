from PCAPrep import PCAprep as PCA
#1. File name 
#2. Column title to be investigated. [chromosome_numbers','genome_size','genome_gc_content', 'protein_coding_genes', 'gene_counts']
#3. Cut off to separate values in gene_count
#4. Threshold to exclude values for component 1. 
#5. Threshold to exclude values for component 2. 

result = PCA('clean.csv', 'gene_counts', 3000, 2, 2)
result.prep()
from PCAOther import PCAlineage as PL
#1. File name
#2 Column title to be investigated. 'sci_name','cultures','lineage', '0', 'complex_medium', 'MediaID'
#3. Rank within lineage (whole integer number)
#4. Search taxa
#4. Threshold to exclude values for component 1. 
#5. Threshold to exclude values for component 2. 
'''result = PL('clean.csv', 'lineage', 0, 'Pseudomonadota',2, 2)
result.prep()'''