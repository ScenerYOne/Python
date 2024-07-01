import pandas as pd
import numpy as np

technologies = {'Courses':["Spark","PySpark","Hadoop","Python","Pandas"],
                'Duration':['35days', '35days', '40days','30days','25days']
                }
df = pd.DataFrame(technologies)
df["Fee"] = np.random.randint(20000, 30000, size=(5, 1))
df["Discount"] = np.random.randint(1000, 3000, size=(5, 1))
print(df)

df["Price"] = np.random.randint(1000, 30000, size=(5, 1))

df1 = df.iloc[:2, :]
df2 = df.iloc[2:, :]

df3 = df.iloc[:, :2]
df4 = df.iloc[:, 2:]

grouped = df1.groupby(df.Duration)
df5 = grouped.get_group("35days")

df6 = df.sample(n=1)
df7 = df.sample(frac=0.5)

print(df1)