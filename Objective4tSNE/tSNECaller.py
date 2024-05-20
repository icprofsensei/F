from tSNEPrep import tSNEprep as tSNE
#1. File name 
#2. Column title to be investigated. [chromosome_numbers','genome_size','genome_gc_content', 'protein_coding_genes', 'gene_counts']
#3. Cut off to separate values in gene_count
#4. Number of groups on the tSNE plot
'''result = tSNE('clean.csv', 'gene_counts', 3000, 2)
result.prep()'''

from tSNEOther import tSNElineage as PL
#1. File name
#2 Column title to be investigated. 'sci_name','cultures','lineage', '0', 'complex_medium', 'MediaID'
#3. Rank within lineage (whole integer number)
#4. Search taxa
#4. Number of groups on the tSNE plot
result = PL('clean.csv', 'lineage', 0, ['Pseudomonadota', 'Myxococcota', 'Terrabacteria', 'FCB'], 5)
result.prep()