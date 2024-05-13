# Looking up the DSMZ id in the mediadive database. eg: JCM identifier --> Scientific name/common name
import json
import csv
import requests
import pandas as pd
from alive_progress import alive_bar
from itertools import islice
with open('mediaperjcmstrain.csv', 'r') as jcms: 
      with open('mediaperjcmspec.csv', 'a+') as new:
            writer = csv.writer(new)
            # Create reader object by passing the file  
            # object to reader method 
            r = csv.reader(jcms)
            # Iterate over each row in the csv  
            # file using reader object 
            print(True)
            for row in r: 
                #print(row)
                if row[0] == '':
                     continue
                else:
                     
                    if int(row[0]) > 3390:
                        #print(True) 
                        spec = row[1]
                        spec = spec.strip('JCM ')
                        #print(spec)
                        url = 'https://mediadive.dsmz.de/rest/strain/' + 'jcm' + '/' + str(spec)
                        response = requests.get(url)
                        if response.status_code == 200:
                            try:

                                resp_dict = response.json()
                                latname = resp_dict['data']['species']
                                row.append(latname)
                                writer.writerow(row)
                            except:
                                print('Could not find exact output')
                                row.append(" ")
                                writer.writerow(row)
                        else:
                            row.append(" ")
                            writer.writerow(row)
                            continue
                    else:
                         continue