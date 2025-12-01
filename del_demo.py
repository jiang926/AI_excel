import pandas as pd

df = pd.read_parquet('/mnt/d/dw/103_2021/20210331/399998', engine='pyarrow')
print(df.head())

df.to_excel('20210331_399998.xlsx', index=False)
