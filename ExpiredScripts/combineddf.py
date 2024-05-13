#Converts from the format of species per medium to medium per species (separated by the different DSMZ prefixes = dsm, jcm, ccap)
import pandas as pd
from alive_progress import alive_bar
df = pd.read_csv("strainspermedium.csv")
#print(df)
species = df['species']
specieslist = []
for i in species:
    i = i.split(',')
    for j in i:
        specieslist.append(j.strip('[]'))
specset = set(specieslist)
speclist = list(specset)
newlist = []
for i in speclist:
    newlist.append(i.replace("'", "").strip())
#print(newlist)

dsm = []
jcm = []
ccap = []
for i in newlist:
    if i.startswith('DSM'):
        dsm.append(i)
    elif i.startswith('JCM'):
        jcm.append(i)
    else:
        ccap.append(i)
#print(dsm)
#print(jcm)
#print(ccap)

specdict = dict.fromkeys(ccap)
with alive_bar(len(ccap)) as bar: 
    for i in specdict:
        
        media = []
        try:

            for index, row in df.iterrows():
                if i in row['species']:
                    media.append(row['medium'])
            specdict[i] = media
            #print(i, media)
        except:
            specdict[i] = []
        bar()
ccapdf = pd.DataFrame(list(specdict.items()))
ccapdf.to_csv('mediaperccapstrain.csv')