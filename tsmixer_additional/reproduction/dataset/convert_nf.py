#%% 
import pandas as pd

df = pd.read_csv('tsmixer/tsmixer_basic/dataset/df_y.csv')
df = df.rename(columns={'ds':'date'})
df = df.pivot(index='date', columns='unique_id', values='y')
df = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']]
df = df.to_csv('tsmixer/tsmixer_basic/dataset/ETTm2_nf.csv')
#%%
