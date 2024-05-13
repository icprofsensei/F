import requests
import pandas as pd
url = 'https://mediadive.dsmz.de/rest/media'

response = requests.get(url)

if response.status_code == 200:

    resp_dict = response.json()
    df = pd.DataFrame(resp_dict.get('data'))
    print(df)
    df.to_csv('medialist.csv')
else:
    print(response.status_code)

