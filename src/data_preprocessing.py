import pandas as pd
df=pd.read_csv(r"C:\Users\hindu\Desktop\car_price linear_regression\data\raw\car_price_dataset.csv")
df=df.drop_duplicates()  #remove duplicates
df=df.dropna() # handle missing values
df.to_csv(r"C:\Users\hindu\Desktop\car_price linear_regression\data\processed\cleaned_car_price_dataset.csv",index=False)
print("Data preprocessing completed")
