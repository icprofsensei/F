# FYP KOMODO. Unrelated to current trajectory. 
import requests
import pandas as pd
response = requests.get('https://komodo.modelseed.org/servlet/KomodoTomcatServerSideUtilitiesModelSeed?OrganismMedia')
print (response.status_code)
a = response.text
b = a.split("\t\t\t\t</TD>\t\t\t\t</TR>\n\t\t\t\t<TR>\n\t\t\t\t\t")
rawhtml = []
for i in b:
    c= i.split("</TD>")
    rawhtml.append(c)
rawhtml.pop(0)
table = []
for rh in rawhtml:
    trow = []
    for rho in range(0, len(rh)):
        cell = rh[rho].split(">")
        
        for i in cell:
            if i.startswith('\n') or i.startswith('<TD') or i.startswith('<a') or i.startswith('|<a') or i.startswith('<td bgcolor'):
                continue
            elif i == ' \t\t\t\t' or i == '\t\t\t\t</TR' or i == '<tr' or i == '</td' or i=='</tr' or i=='</table' or i==" ":
                continue
            else:
                i = i.rstrip('</a')
                trow.append(i)
        
     # Step to ensure that all rows have 4 columns to fit into dataframe
    table.append(trow[:4])
    #print(len(trow))
    
#print(table)       
#print(rawhtml)
df = pd.DataFrame(table, columns = ["Organism DSMZ ID" , "TaxonID", "Organism Name", "Media list"])
print(df)
df.to_csv('KOMODO.csv')