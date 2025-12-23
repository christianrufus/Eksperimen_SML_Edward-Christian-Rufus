import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cfo = fetch_california_housing()
data = pd.DataFrame(cfo.data, columns=cfo.feature_names)
data['target'] = cfo.target

def outlier_clearing(data, y):
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1
  cond = ~(
    (data < (Q1 - 1.5 * IQR)) |
    (data > (Q3 + 1.5 * IQR))
    ).any(axis=1)
  df_neat = data.loc[cond, data.columns]

  y = y.loc[df_neat.index]
  y.head()
  return df_neat, y

def auto_data(data):
  df = data.copy()
  # Drop kolom tidak penting
  df = df.drop(columns=['AveBedrms', 'AveOccup', 'Population'], axis=1)
  # Outlier handling
  df_neat, y = outlier_clearing(df, data['target'])
  # Splitting
  X_train, X_test, y_train, y_test = train_test_split(df_neat, y, test_size=0.2, random_state=13)
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = auto_data(data)
X_train.to_csv("Xtr.csv", index=False)
X_test.to_csv("Xte.csv", index=False)
y_train.to_csv("ytr.csv", index=False)
y_test.to_csv("yte.csv", index=False)

