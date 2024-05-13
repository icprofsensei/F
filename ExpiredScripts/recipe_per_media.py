# Use list of media (media_0.json) to API call the recipe and required ingredients (+ their quantity). Fill in a ingredients dictionary (based on ingredients_0.json) and append to the dataframe.
import json
import requests
import pandas as pd
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
for ml in medialist:
    print(ml)
    url = 'https://mediadive.dsmz.de/download/medium/' + str(ml) + '/json'
    response = requests.get(url)
    if response.status_code == 200:
        try:

                resp_dict = response.json()
                recipe = resp_dict["solutions"][0]["recipe"]
                clean_recipe = dict.fromkeys(ingredientslist)
                for i in recipe: 
                    if i['optional'] == 'no' and (i['unit'] == 'g' or i['unit'] == 'g/l'):
                        clean_recipe[i['compound_id']] = i['g_l']
                    elif i['optional'] == 'no' and i['unit'] == 'ml':
                        if 'compound' in i.keys():
                            clean_recipe[i['compound_id']] = i['amount']
                        elif 'solution' in i.keys():
                            clean_recipe[i['solution_id']] = i['amount']
                    else:
                        continue
            
                data.append(clean_recipe)
        except:
            clean_recipe = dict.fromkeys(ingredientslist)
            data.append(clean_recipe)
df = pd.DataFrame(data, index = medialist)
print(df)
df.to_csv('mediaingredients.csv')