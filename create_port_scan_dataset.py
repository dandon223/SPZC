import pandas as pd
import glob

# Zbior danych CICIDS2017 z https://www.kaggle.com/datasets/cicdataset/cicids2017?resource=download

files = glob.glob("MachineLearningCVE/*.csv")
df = pd.concat(map(pd.read_csv, files), ignore_index=True)

print(df.shape)
df.dropna(inplace=True)
df = df.loc[df[' Label'].isin(['BENIGN', 'PortScan'])]


print(df.shape)
df.to_csv("MachineLearningCVE/PortScan1.csv")
