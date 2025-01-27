import numpy as np
import os 
from pathlib import Path 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_iris_dataframe():
    CSV_PATH = Path(os.path.dirname(__file__)) / "iris.csv"
    return pd.read_csv(CSV_PATH)

    

def get_binary_iris_dataset(viz = False , verbose = False ):
    df = get_iris_dataframe()
    if viz:
        df['petal.length'].hist()
        df['petal.width'].hist()
        plt.show()


    # encode 
    le = LabelEncoder()
    df['variety'] = le.fit_transform(df['variety'])
    df = df[df['variety'] != 0]
    if viz:
        df['petal.length'].hist()
        df['petal.width'].hist()

    # shuffle df
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.drop(columns = ['variety'])
    Y = df['variety']
    X = X.to_numpy()
    Y = Y.to_numpy() -1

    print(f"Y set - {set(Y)}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state=0)
    return X_train, Y_train, X_test, Y_test