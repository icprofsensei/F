import pandas as pd
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
clean = pd.read_csv('base.csv')
alltaxids = list(set([str(row['taxid']) for index, row in clean.iterrows()]))
nwdict = {key: [] for key in alltaxids}
inghead = list(clean.columns.values)[7:]
for index, row in clean.iterrows():
    id = row['taxid']
    req_res = []
    for index2, r in enumerate(row[7:]):
        if r!= 0:
            req_res.append(inghead[index2])
        else:
            continue
    for i in req_res:
        if i not in nwdict[str(id)]:
            nwdict[str(id)].append(i)
        else:
            continue

df = pd.DataFrame(nwdict.items())
newcol = []
for index, row in df.iterrows():
    resources = literal_eval(str(row[1]))
    nichewidth = len(resources)
    newcol.append(nichewidth)
df.insert(2, 'Niche_Width', newcol, True)
df.to_csv('check.csv')
counts, bins = np.histogram(np.log10(df['Niche_Width']), bins = 'fd')
print(bins)
print(len(bins))
plt.stairs(counts, bins, color = '#ecec5f',fill = True )
plt.title('Histogram of prokaryotic resource niche widths', fontsize = 18)
plt.xlabel('Log10 (Niche width)', fontsize = 18)
plt.ylabel('Number', fontsize = 18)
plt.show()