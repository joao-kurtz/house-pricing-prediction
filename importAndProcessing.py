import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

file_path = 'data/housing.csv'

def importAndProcessing():

    file_path = 'data/housing.csv'

    
    df = pd.read_csv(file_path)

    #briging price into millions scale
    df['price'] = df['price']/1000000
    #outliers handling
    Q1_price = np.percentile(df['price'], 25)
    Q3_price = np.percentile(df['price'], 75)
    IQR_price = Q3_price - Q1_price

    upper_bound_price = Q3_price + 1.5 * IQR_price
    outliers_price = df['price'] > upper_bound_price

    Q1_area = np.percentile(df['area'], 25)
    Q3_area = np.percentile(df['area'], 75)
    IQR_area = Q3_area - Q1_area

    upper_bound_area = Q3_area + 1.5 * IQR_area

    outliers_area = df['area'] > upper_bound_area

    df = df[~(outliers_price | outliers_area)]

    #data preparation

    dummyVariables = ['mainroad','guestroom','basement', 'hotwaterheating','airconditioning', 'prefarea', 'furnishingstatus']
    num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    #creating dummy variable for categorical columns
    dummy=pd.get_dummies(df[dummyVariables],drop_first=True)
    dummy=dummy.astype('int')
    df=df.drop(columns=dummyVariables,axis=1)
    df=pd.concat([df,dummy], axis=1)

    x=df.drop('price', axis=1)
    y=df['price']

    #Scaling numeric variables
    x_numeric = x[num_vars]
    sc=RobustScaler()
    x_numeric_scaled=sc.fit_transform(x_numeric)
    x[num_vars] = x_numeric_scaled

    return x, y

print("script executado")