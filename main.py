import numpy as np
import pandas as pd
import pickle
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot
import datetime


#impoorting the data
data=pd.read_csv('/Users/amyqk/Downloads/Liver Patient Dataset (LPD)_train.csv',encoding = 'unicode_escape')
#getting rid of rows with blank values
new_data = data.dropna()
display(new_data.head())
new_data.info()

#replacing males with value 0 and females with value 1
pd.options.mode.chained_assignment = None
new_data['Gender of the patient'] = new_data['Gender of the patient'].replace(['Male'], '0')
new_data['Gender of the patient'] = new_data['Gender of the patient'].replace(['Female'], '1')
display(new_data["Gender of the patient"].head())
new_data.info()

#isolating the x (independent variables)
X=new_data.drop('Result',axis=1)
display(X.head())
display(X.shape)

#isolating the y (dependent variable)
y=new_data.Result
display(y.head())
display(y.shape)

new_data_list = list(X.columns)
#print(new_data_list)
#new_data = np.array(new_data)


#graph of the current data results
y.value_counts()
sns.countplot(x="Result",data=new_data,palette='hls')
plt.show()
plt.savefig('count_plot')


#split data into training (80%) and testing (20%)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)
print("X_train : ",X_train.shape)
print("X_test : ",X_test.shape)
print("y_train : ",y_train.shape)
print("y_test : ",y_test.shape)

def linear_regression():
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)

    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)
    lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']
    print(lr_results)
    print(lr.score(X_test,y_test))

def random_forest():
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)

    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)

    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)


    rf_results = pd.DataFrame(['Random forest',rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
    rf_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']
    print(rf_results)
    print(rf.score(X_test,y_test))

    filename = 'rf_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    #print(result)

    """
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    export_graphviz(tree, out_file = 'tree.dot', feature_names = new_data_list, rounded = True, precision = 1)
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')
    """
"""
# Get numerical feature importances
importances = list(loaded_model.new_data_importances)
# List of tuples with variable and importance
new_data_importances = [(feature, round(importance, 2)) for feature, importance in zip(new_data_list, importances)]
# Sort the feature importances by most important first
new_data_importances = sorted(new_data_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in new_data_importances];
"""
"""
# Dates of training values
months = new_data[:, new_data_list.index('month')]
days = features[:, new_data_list.index('day')]
years = features[:, new_data_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60');
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
"""
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
	keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=34, batch_size=1, validation_data=(x_test, y_test))

loss_train = history.history['train_loss']
loss_val = history.history['val_loss']
epochs = range(1,35)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
"""

if __name__ == '__main__':
    random_forest()