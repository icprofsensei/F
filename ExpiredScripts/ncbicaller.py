#For every species in cleaned.csv, first search if it is a prokaryote (via the taxonomy API call) and if so, make a second genomic API call to obtain intrinsic information. 
import csv
from ast import literal_eval
import requests
import json
from alive_progress import alive_bar
ncbikey = 'f55726c2c32772c2b82304814b30148aff07'
species = []
rows = []
with open('cleaned.csv', 'r') as old: 
    
    r = csv.reader(old)
    for row in r:
            species.append(row[0])
            rows.append(row)
prokkingdic = {'2': 'bacteria', '2157':'archaea'}
with open('prokdata.csv', 'a+') as pd:
    csvwriter = csv.writer(pd)
    csvwriter.writerow(['name','cultures','taxid','sci_name','chromosome_numbers','genome_size', 'genome_gc_content', 'gene_counts','protein_coding_genes'])
with alive_bar(len(species)) as bar: 
    for i in range(17711, len(species)):
                        with open('problemtaxa.txt', 'a+') as pt:
                            with open('prokdata.csv', 'a+') as pd:
                                csvwriter = csv.writer(pd)
                                spec = species[i]
                                print(spec)
                                prok = False
                                url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy/taxon/" + str(spec)
                                response = requests.get(url, params =  {"key": ncbikey})
                                if response.status_code == 200:
                                    #print(spec)
                                    try:
                                        resp_dict = response.json()
                                        lineage = resp_dict['taxonomy_nodes'][0]['taxonomy']['lineage']
                                        for l in lineage:
                                                if str(l) in prokkingdic.keys():
                                                        prok = True
                                                else:
                                                        continue
                                        if prok == True:
                                                    url2 = "https://api.ncbi.nlm.nih.gov/datasets/v1/genome/taxon/" +  str(spec) + "?filters.reference_only=true&filters.assembly_level=chromosome&filters.assembly_level=complete_genome"
                                                    response = requests.get(url2, params =  {"key": ncbikey})
                                                    if response.status_code == 200:
                                                        try:
                                                                
                                                                    resp_dict = response.json()
                                                                    
                                                                    chromosomes = resp_dict['assemblies'][0]['assembly']['chromosomes']
                                                                    organism = resp_dict['assemblies'][0]['assembly']['org']
                                                                    gene_counts = resp_dict['assemblies'][0]['assembly']['annotation_metadata']['stats']['gene_counts']
                                                                    if len(chromosomes) == 1:
                                                                            
                                                                        if gene_counts == 'missing':
                                                                                dictsp = {'name': rows[i][0], 'cultures': rows[i][1], 'taxid':organism['tax_id'], 'sci_name': organism['sci_name'], 'chromosome_count': len(chromosomes), 'genome_length': chromosomes[0]['length'], 'genome_gc': chromosomes[0]['gc_count'], 'gene_counts': 'N/A', 'protein_coding_genes': 'N/A'}
                                                                                csvwriter.writerow(dictsp.values())
                                                                        else:
                                                                                
                                                                                dictsp = {'name': rows[i][0], 'cultures': rows[i][1], 'taxid':organism['tax_id'], 'sci_name': organism['sci_name'], 'chromosome_count': len(chromosomes), 'genome_length': chromosomes[0]['length'], 'genome_gc': chromosomes[0]['gc_count'], 'gene_counts': gene_counts['total'], 'protein_coding_genes': gene_counts['protein_coding']}
                                                                                csvwriter.writerow(dictsp.values())
                                                                    elif len(chromosomes) > 1:
                                                                        length = [int(i['length']) for i in chromosomes]
                                                                        gc = [int(i['gc_count']) for i in chromosomes]
                                                                        if gene_counts == 'missing':
                                                                                dictsp = {'name': rows[i][0], 'cultures': rows[i][1], 'taxid':organism['tax_id'], 'sci_name': organism['sci_name'], 'chromosome_count': len(chromosomes), 'genome_length': sum(length), 'genome_gc': sum(gc), 'gene_counts': 'N/A', 'protein_coding_genes': 'N/A'}
                                                                                csvwriter.writerow(dictsp.values())
                                                                        else:
                                                                                
                                                                                dictsp = {'name': rows[i][0], 'cultures': rows[i][1], 'taxid':organism['tax_id'], 'sci_name': organism['sci_name'], 'chromosome_count': len(chromosomes), 'genome_length': sum(length), 'genome_gc': sum(gc), 'gene_counts': gene_counts['total'], 'protein_coding_genes': gene_counts['protein_coding']}
                                                                                csvwriter.writerow(dictsp.values())
                                                                        '''
                                                                        length = 0
                                                                        gc = 0
                                                                        for i in chromosomes:
                                                                            length += int(str(i['length']))
                                                                            gc += int(str(i['gc_count']))
                                                                        if gene_counts == 'missing':
                                                                                dictsp = {'name': rows[i][0], 'cultures': rows[i][1], 'taxid':organism['tax_id'], 'sci_name': organism['sci_name'], 'chromosome_count': len(chromosomes), 'genome_length': 'length', 'genome_gc': gc, 'gene_counts': 'N/A', 'protein_coding_genes': 'N/A'}
                                                                                csvwriter.writerow(dictsp.values())
                                                                        else:
                                                                                
                                                                                dictsp = {'name': rows[i][0], 'cultures': rows[i][1], 'taxid':organism['tax_id'], 'sci_name': organism['sci_name'], 'chromosome_count': len(chromosomes), 'genome_length': length, 'genome_gc': gc, 'gene_counts': gene_counts['total'], 'protein_coding_genes': gene_counts['protein_coding']}
                                                                                csvwriter.writerow(dictsp.values())       
                                                                        '''
                                                        except:
                                                            #print('Missing data entries for metadata')
                                                            pt.write(spec + ' Missing data entries for metadata' + '\n')

                                                            continue
                                                    else:
                                                        #print('No metadata lookup', response.status_code)
                                                        pt.write(spec + ' No metadata lookup'+ '\n')
                                                        continue
                                        else:
                                                #print('Eukaryote')
                                                pt.write(spec + ' Eukaryote'+ '\n')
                                    except:
                                            #print('Missing data entries for lineage')
                                            pt.write(spec + ' Missing data entries for lineage'+ '\n')
                                            continue
                                else:
                                    #print('No lineage look up', response.status_code)
                                    pt.write(spec + ' No lineage look up'+ '\n')
                                    continue
    bar()
                                                    
                                                
                                    