import pandas as pd
data = {
    'cancer patient name' : ['raju','rani','ravi','ramu'],
    'cancer' : ['yes','no','no','yes']
}

# converting data set to dataframe:
df = pd.DataFrame(data)
print(df)
df.to_csv('cancer.csv')