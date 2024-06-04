import pandas as pd
import matplotlib.pyplot as plt
# Sample DataFrame
data = {'cat1': [1, 7, 5, 4, 6, 8, 3, 9, 2, 10], 
        'cat2': [0.51, 0.37, 0.65, 0.74, 0.96, 0.28, 0.43, 0.19, 0.22, 0.10]}
df = pd.DataFrame(data)
plt.scatter(data['cat1'], data['cat2'])
plt.show()
# Create bins based on cat2
df['bins'] = pd.qcut(df['cat2'], q=4)

# Function to rank 'cat1' within each bin and drop lower 50% of each bin
def drop_lower_50(df_group):
    df_group['rank'] = df_group['cat1'].rank()
    df_group = df_group[df_group['rank'] > df_group['rank'].median()]  # Drop lower 50% of data in each bin
    return df_group

# Apply the function to each group in 'bins'
df_filtered = df.groupby('bins').apply(drop_lower_50)

# Reset index
df_filtered.reset_index(drop=True, inplace=True)

# Drop the 'bins' column as it's no longer needed
df_filtered.drop('bins', axis=1, inplace=True)

# Output the filtered DataFrame
print(df_filtered)
plt.scatter(df_filtered['cat1'], df_filtered['cat2'])
plt.show()