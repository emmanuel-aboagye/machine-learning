# -*- coding: utf-8 -*-
"""
@author: aboag
"""
#%% Install hyperopt
#pip install hyperopt

#%% Import modules
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, HeNormal
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
from sklearn.inspection import permutation_importance
#%%
# set random seed for reproducability
SEED=0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.experimental.numpy.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

#%% Define functions
# function to import data
def import_data(path):
    raw_data = pd.read_csv(path, header=None, na_values='?')
    raw_X_data = raw_data.iloc[:,1:23]
    raw_y_data = raw_data.iloc[:,23:30]
    data = pd.concat([raw_X_data,raw_y_data], axis=1)
    old_cols = [ name for name in data.columns]
    new_cols = ['flowrate_input','zinc_input','pH_input','BOD_input','COD_input','TSS_input','VSS_input',
                 'sediments_input','conductivity_input','pH_primary_settler_input','BOD_primary_settler_input','TSS_primary_settler_input',
                 'VSS_primary_settler_input','sediments_primary_settler_input','conductivity_primary_settler_input',
                 'pH_secondary_settler_input','BOD_secondary_settler_input','COD_secondary_settler_input',
                 'TSS_secondary_settler_input','VSS_secondary_settler_input','sediments_secondary_settler_input',
                 'conductivity_secondary_settler_input','pH_output','BOD_output','COD_output','TSS_output','VSS_output',
                 'sediments_output','conductivity_output',
               ]
    cols_dict = dict(zip(old_cols, new_cols))
    data = data.rename(columns=cols_dict)
    data.dropna(how='any', axis=0, inplace=True)
    return data

# function to scale features
def feature_standardization(feature_set):
    scalar = StandardScaler()
    scaled_feature_set = scalar.fit_transform(feature_set)
    return scaled_feature_set

# function to split data into train-validation-test
def data_splitting(feature_set,label_set,ts=0.3,rs=42):
    feature_train, feature_, label_train, label_ = train_test_split(feature_set, label_set, test_size=ts, random_state=rs)
    feature_val, feature_test, label_val, label_test = train_test_split(feature_, label_, test_size=0.5, random_state=rs)
    return [feature_train, feature_val, feature_test, label_train, label_val, label_test]

# function for nueral network model
def nn_model(feature_train):
    tf.keras.utils.set_random_seed(SEED)
    model = Sequential()
    model.add(Dense(units=60, input_dim=feature_train.shape[1], activation='elu', kernel_initializer=GlorotUniform(seed=0)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=40, kernel_initializer=GlorotUniform(seed=0), activation=LeakyReLU(alpha=0.05)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=20, kernel_initializer=GlorotUniform(seed=0), activation=LeakyReLU(alpha=0.05)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=1, activation='relu', kernel_initializer=GlorotUniform(seed=0)))
    model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(learning_rate=0.01))
    return model

# function to make predictions 
def make_predictions(model, feature_train, feature_val, feature_test):
    label_train_pred = model.predict(feature_train)
    label_val_pred = model.predict(feature_val)
    label_test_pred = model.predict(feature_test)
    return [label_train_pred, label_val_pred, label_test_pred]

# function for model evaluation
def model_evaluation(label_train,label_train_pred,label_val,label_val_pred,label_test,label_test_pred):
    rmse_train = np.sqrt(mean_squared_error(label_train, label_train_pred))
    rmse_val   = np.sqrt(mean_squared_error(label_val, label_val_pred))
    rmse_test  = np.sqrt(mean_squared_error(label_test, label_test_pred))
    r2_score_train = r2_score(label_train, label_train_pred)
    r2_score_val   = r2_score(label_val, label_val_pred)
    r2_score_test  = r2_score(label_test, label_test_pred)
    print('RMSE: train=', rmse_train, 'validation=', rmse_val, 'test=', rmse_test)
    print('r2_score: train=',r2_score_train, 'validation=', r2_score_val, 'test=', r2_score_test)
    return [rmse_train, rmse_val, rmse_test, r2_score_train, r2_score_val, r2_score_test]

# function for plotting
def plot_results(label_train,label_train_pred,label_val,label_val_pred,label_test,label_test_pred,r2_score_train,r2_score_val,r2_score_test):
    plt.figure(figsize=(7,5), dpi=700)
    plt.plot(label_train, label_train_pred, 'bo', label=f'train: r-sqaured={r2_score_train:.3f}')
    plt.plot(label_val, label_val_pred, 'yo', label=f'valid: r-sqaured={r2_score_val:.3f}')
    plt.plot(label_test, label_test_pred, 'ro', label=f'test:  r-sqaured={r2_score_test:.3f}')
    plt.plot([label_train.min(), label_train.max()], [label_train.min(), label_train.max()], 'k--')
    plt.xlabel('Output Conductivity (Actual)')
    plt.ylabel('Output Conductivity, (Predicted)')
    plt.title('Prediction of Conductivity for Wastewater Treatment Plant')
    plt.legend()
    plt.show()
    
#%% Import data
path = 'water-treatment.data'
data = import_data(path)

#%% Organze data into features and label
feature_set = data.iloc[:,:22]
label_set = data['conductivity_output']

#%% Scale data and split into train-validation-test
scaled_feature_set = feature_standardization(feature_set)
feature_train, feature_val, feature_test, label_train, label_val, label_test = data_splitting(scaled_feature_set, label_set, ts=0.3, rs=42)

#%% fit model 
model = nn_model(feature_train)
model.fit(feature_train, label_train, epochs=4000, batch_size=0, verbose=1, validation_data=(feature_val,label_val))

#%% make some predictions
label_train_pred, label_val_pred, label_test_pred = make_predictions(model, feature_train, feature_val, feature_test)

#%% check some metrics 
rmse_train,rmse_val,rmse_test,r2_score_train,r2_score_val,r2_score_test =  model_evaluation(label_train,label_train_pred,label_val,label_val_pred,label_test,label_test_pred)

#%% plotting
plot_results(label_train,label_train_pred,label_val,label_val_pred,label_test,label_test_pred,r2_score_train,r2_score_val,r2_score_test)

#%% Use hyperopt to tune hyperparameters 
# Define search space and function for hyperparameter tuning using 'hyperopt'
space = {
         'learning_rate': hp.uniform('learning_rate', 0.001,0.1),
         'loss' : hp.choice('loss', ['mean_squared_error', 'mean_absolute_error', 'logcosh', 'mean_squared_logarithmic_error']),
         'epochs': scope.int(hp.choice('epochs', range(500,5000,500))),
         'batch_size': scope.int(hp.quniform('batch_size',0,250,25)),
         'alpha': hp.uniform('alpha', 0.001,0.3),
         'activation' : hp.choice('activation',['relu','tanh','elu', 'sigmoid']),
         'rate' : hp.uniform('rate',0.001,0.5),
         'rate1'  : hp.uniform('rate1',0.001,0.5),
         'units'  : scope.int(hp.quniform('units',10,150,10)),
         'units1' : scope.int(hp.quniform('units1',10,150,10)),
         'units2' : scope.int(hp.quniform('units2',10,150,10)),
}

# define an objective function
def ann_model_hppt(space): 
    tf.keras.utils.set_random_seed(SEED)
    model = Sequential()
    model.add(Dense(units=space['units'], input_dim=feature_train.shape[1], activation=space['activation'], kernel_initializer=GlorotUniform(seed=0)))
    model.add(Dropout(rate=space['rate']))
    model.add(Dense(units=space['units1'], kernel_initializer=GlorotUniform(seed=0), activation=LeakyReLU(alpha=space['alpha'])))
    model.add(Dropout(rate=space['rate1']))
    model.add(Dense(units=space['units2'], kernel_initializer=GlorotUniform(seed=0), activation=LeakyReLU(alpha=space['alpha'])))
    model.add(Dropout(rate=space['rate1']))
    model.add(Dense(units=1, activation='relu', kernel_initializer=GlorotUniform(seed=0)))
    model.compile(loss=space['loss'], optimizer=Adam(learning_rate=space['learning_rate']))
    model.fit(feature_train, label_train, epochs=space['epochs'], batch_size=space['batch_size'], verbose=0, validation_data=(feature_val,label_val))
    label_pred_val = model.predict(feature_val)
    label_pred_train = model.predict(feature_train)
    score = np.sqrt(mean_squared_error(label_val, label_pred_val)) + np.sqrt(mean_squared_error(label_train, label_pred_train)) * 0.5   # just a trick I use to improve my model 
    return {'loss': score, 'status': STATUS_OK}

#%% Hyperparameters tuning 
trials = Trials()
Best = fmin(fn=ann_model_hppt,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print('Best hyperparameters: ', space_eval(space,Best))
print('Best loss: ', trials.best_trial['result']['loss'])

#%% Save results from hyperparameter tuning in a text file (Optional)
with open("hppt_ann_model.txt", "w") as file:
    file.writelines("Best loss value: " + str(trials.best_trial['result']['loss']))
    file.write("\n------------------------------------------------\n")
    file.writelines("Best hyperparameters: " + str(space_eval(space,Best)))
    file.write("\n------------------------------------------------\n")
file.close()

#%%
'''{'activation': 'elu', 'alpha': 0.13483780311427757, 'batch_size': 100, 'epochs': 4000, 'learning_rate': 0.0018902548925562176, 
 'loss': 'logcosh', 'rate': 0.13143546639637724, 'rate1': 0.06138740290792727, 'units': 110, 'units1': 120, 'units2': 130}
'''

#%% recreate the new architechture with the new hyperparamters and refit the model
def ann_model_tune(feature_train): 
    tf.keras.utils.set_random_seed(SEED)
    model = Sequential()
    model.add(Dense(units=110, input_dim=feature_train.shape[1], activation='elu', kernel_initializer=GlorotUniform(seed=0)))
    model.add(Dropout(rate=0.13143546639637724))
    model.add(Dense(units=120, kernel_initializer=GlorotUniform(seed=0), activation=LeakyReLU(alpha=0.13483780311427757)))
    model.add(Dropout(rate=0.06138740290792727))
    model.add(Dense(units=130, kernel_initializer=GlorotUniform(seed=0), activation=LeakyReLU(alpha=0.13483780311427757)))
    model.add(Dropout(rate=0.06138740290792727))
    model.add(Dense(units=1, activation='relu', kernel_initializer=GlorotUniform(seed=0)))
    model.compile(loss='logcosh', optimizer=Adam(learning_rate=0.0018902548925562176))
    return model

model_hppt = ann_model_tune(feature_train)
model_hppt.fit(feature_train, label_train, epochs=4000, batch_size=100, verbose=1, validation_data=(feature_val,label_val))

#%% make some predictions
label_train_pred_hppt, label_val_pred_hppt, label_test_pred_hppt = make_predictions(model_hppt, feature_train, feature_val, feature_test)

#%% check some metrics 
rmse_train_hppt,rmse_val_hppt,rmse_test_hppt,r2_score_train_hppt,r2_score_val_hppt,r2_score_test_hppt =  model_evaluation(label_train,label_train_pred_hppt,label_val,label_val_pred_hppt,label_test,label_test_pred_hppt)

#%% plotting
plot_results(label_train,label_train_pred_hppt,label_val,label_val_pred_hppt,label_test,label_test_pred_hppt,r2_score_train_hppt,r2_score_val_hppt,r2_score_test_hppt)

#%% Feature importance
scorer = make_scorer(mean_squared_error, greater_is_better=False)
perm = permutation_importance(model_hppt, feature_test, label_test, n_repeats=2, random_state=42, scoring=scorer)
importances = perm.importances_mean
idxx = np.argsort(importances)
#%% plot importance
plt.figure(figsize=(10,7), dpi=700)
plt.barh(range(len(idxx)),importances[idxx])
plt.xlabel("Nerual Network Model: Feature Importance")
plt.yticks(range(len(idxx)), [feature_set.columns[i] for i in idxx])
plt.show()

#%%
'''
Ideally, you would want to tune your model first and then create the architechture from the optimized hyperparamters

You can also refit the model using the, probably, top 5 features with the highest importances. 
'''
