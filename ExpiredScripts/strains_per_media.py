# Species associated with each medium
import pandas as pd
import json
import requests
df = pd.read_csv("mediaingredients.csv")
#print(df)
Mediatype = df['Media']
Mediatype = list(Mediatype)
#print(Mediatype)
mts = []
data = []
for mt in Mediatype:
    print(mt)
    url = 'https://mediadive.dsmz.de/rest/medium-strains/' + str(mt)
    response = requests.get(url)
    if response.status_code == 200:
        try:
                mts.append(mt)
                speclist = []
                resp_dict = response.json()
                #print(resp_dict)
                for d in resp_dict['data']:
                     speclist.append(d['ccno'])
                speclist = set(speclist)
                speclist = list(speclist)
                entry = {'species': speclist}
                #print(entry, type(entry))
                data.append(entry)
        except:
            print("Error")
    else:
         print("Fail")
df = pd.DataFrame(data, index = mts)
print(df)
df.to_csv('strainspermedium.csv')