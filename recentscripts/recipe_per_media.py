# Use list of media (media_0.json) to API call the recipe and required ingredients (+ their quantity). Fill in a ingredients dictionary (based on ingredients_0.json) and append to the dataframe.
import json
import requests
import pandas as pd
from alive_progress import alive_bar
#list of media
opendic = open('media_0.json', encoding = 'utf-8') 
dict_data = json.load(opendic)
medialist = [i['medium_id'] for i in dict_data]
#print(medialist)
#list of ingredients
opendic2 = open('ingredients_0.json', encoding = 'utf-8') 
dict_data2 = json.load(opendic2)
ingredientslist = [i['id'] for i in dict_data2]
#print(ingredientslist)
data = []
indexlist = []
with alive_bar(len(medialist)) as bar: 
    for ml in medialist:
        #print(ml)
        url = 'https://mediadive.dsmz.de/download/medium/' + str(ml) + '/json'
        response = requests.get(url)
        if response.status_code == 200:
            indexlist.append(ml)
            try:

                    resp_dict = response.json()
                    recipe = resp_dict["solutions"][0]["recipe"]
                    clean_recipe = dict.fromkeys(ingredientslist)
                    for i in recipe: 
                        if i['optional'] == 'no' and (i['unit'] == 'g'):
                            clean_recipe[i['compound_id']] = str(i['g_l']) + 'g'
                        elif i['optional'] == 'no' and i['unit'] == 'g/l':
                            clean_recipe[i['compound_id']] = str(i['g_l']) + 'g/l'
                        elif i['optional'] == 'no' and i['unit'] == 'ml':
                            if 'compound' in i.keys():
                                clean_recipe[i['compound_id']] = str(i['amount']) + 'ml'
                            elif 'solution' in i.keys():
                                clean_recipe[i['solution_id']] = str(i['amount']) + 'ml'
                        else:
                            continue
                    clean_recipe['min_pH'] = resp_dict["medium"]["min_pH"]
                    clean_recipe['max_pH'] = resp_dict["medium"]["max_pH"]
                    clean_recipe['complex_medium'] = resp_dict["medium"]["complex_medium"]
                    data.append(clean_recipe)
                    
            except:
                clean_recipe = dict.fromkeys(ingredientslist)
                clean_recipe['min_pH'] = ""
                clean_recipe['max_pH'] = ""
                clean_recipe['complex_medium'] = ""
                data.append(clean_recipe)
        bar()
df = pd.DataFrame(data, index = indexlist)
print(df)
df.to_csv('mediaingredients_complete.csv')