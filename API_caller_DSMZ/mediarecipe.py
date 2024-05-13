import requests
import pandas as pd
import json
url = 'https://mediadive.dsmz.de/download/medium/1/json'

response = requests.get(url)

if response.status_code == 200:
    resp_dict = response.json()
    format = json.dumps(resp_dict, indent = 4)
    recipe = resp_dict["solutions"][0]["recipe"]
    clean_recipe = []
    for i in recipe: 
        if i['optional'] == 'no' and (i['unit'] == 'g' or i['unit'] == 'g/l'):
            list = [i['compound'], i['compound_id'], i['g_l']]
            clean_recipe.append(list)
        elif i['optional'] == 'no' and i['unit'] == 'ml':
            if 'compound' in i.keys():
                list = [i['compound'], i['compound_id'], i['amount']]
            elif 'solution' in i.keys():
                list = [i['solution'], i['solution_id'], i['amount']]
            
            clean_recipe.append(list)
        else:
            continue
    print(clean_recipe)
else:
    print(response.status_code)

