import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)

print(df["calories"])
print(df.describe())
print(df.loc[0])
print(df.loc[[0, 1]])