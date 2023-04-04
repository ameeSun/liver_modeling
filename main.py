import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('/Users/amyqk/Downloads/Liver Patient Dataset (LPD)_train.csv',encoding = 'unicode_escape')
new_data = data.dropna()
display(new_data.head())
new_data.info()

#data_cols = ["Age of the patient","Gender of the patient","Total Bilirubin","Direct Bilirubin"," Alkphos Alkaline Phosphotase"," Sgpt Alamine Aminotransferase","Sgot Aspartate Aminotransferase","Total Protiens"," ALB Albumin","A/G Ratio Albumin and Globulin Ratio"]
x=new_data.drop('Result',axis=1)
display(x.head())
display(x.shape)

y=new_data.Result
display(y.head())
display(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
print("X_train : ",x_train.shape)
print("X_test : ",x_test.shape)
print("y_train : ",y_train.shape)
print("y_test : ",y_test.shape)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']