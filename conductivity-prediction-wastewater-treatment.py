# -*- coding: utf-8 -*-
"""
@author: aboag
"""
#%% Install hyperopt
#pip install hyperopt

#%% Import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

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

# function for xgboost model
def xgb_model(feature_train, feature_val, label_train, label_val, lr=0.1, rs=42, vb=0, n_est=100):
    model = model = XGBRegressor(n_estimators=n_est,learning_rate=lr, random_state=rs)
    model.fit(feature_train, label_train, eval_set=[(feature_val,label_val)], verbose=vb)
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
model = xgb_model(feature_train, feature_val, label_train, label_val, lr=0.1, rs=42, vb=0, n_est=100)

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
         'n_estimators': hp.choice('n_estimators',range(10,150,10)),
}

# define an objective function
def xgb_model_hppt(space): 
  model = XGBRegressor(objective='reg:squarederror',
                       n_estimators=space['n_estimators'],
                       learning_rate=space['learning_rate'],
                       verbosity=0,
                       random_state=42)
  model.fit(feature_train,label_train,eval_set=[(feature_val,label_val)],verbose=False)
  label_pred_val = model.predict(feature_val)
  score = np.sqrt(mean_squared_error(label_val, label_pred_val))
  return {'loss': score, 'status': STATUS_OK}

#%% Hyperparameters tuning 
trials = Trials()
Best = fmin(fn=xgb_model_hppt,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials)
print('Best hyperparameters: ', space_eval(space,Best))
print('Best loss: ', trials.best_trial['result']['loss'])

#%% Save results from hyperparameter tuning in a text file (Optional)
with open("hppt_xgb_model.txt", "w") as file:
    file.writelines("Best loss value: " + str(trials.best_trial['result']['loss']))
    file.write("\n------------------------------------------------\n")
    file.writelines("Best hyperparameters: " + str(space_eval(space,Best)))
    file.write("\n------------------------------------------------\n")
file.close()

#%% refit the model with the new hyperparamters 
# Best hyperparameters:  {'learning_rate': 0.09301428684592522, 'n_estimators': 60} ; Best loss:  117.29775645331853
model_hppt = xgb_model(feature_train, feature_val, label_train, label_val, lr=0.09301428684592522, rs=42, vb=0, n_est=60)

#%% make some predictions
label_train_pred_hppt, label_val_pred_hppt, label_test_pred_hppt = make_predictions(model_hppt, feature_train, feature_val, feature_test)

#%% check some metrics 
rmse_train_hppt,rmse_val_hppt,rmse_test_hppt,r2_score_train_hppt,r2_score_val_hppt,r2_score_test_hppt =  model_evaluation(label_train,label_train_pred_hppt,label_val,label_val_pred_hppt,label_test,label_test_pred_hppt)

#%% plotting
plot_results(label_train,label_train_pred_hppt,label_val,label_val_pred_hppt,label_test,label_test_pred_hppt,r2_score_train_hppt,r2_score_val_hppt,r2_score_test_hppt)

#%% Feature importance
sorted_idx = model_hppt.feature_importances_.argsort()
plt.figure(figsize=(10,7), dpi=700)
plt.barh(feature_set.columns[sorted_idx], model_hppt.feature_importances_[sorted_idx])
plt.xlabel("XGBoost Model: Feature Importance")
plt.show()

#%% Comments

"""
After tuning the learning rate (also known as step-size in engineering) and the number of trees, the model 
seems to be generalizing well. 

The 'conductivity_output' label can be changed with any desired output label for model building and prediction

There are several hyperparameters associated with XGBoost which can be tuned to improve the model generalization. 
The some important ones are:
max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, num_leaves, boosting_type

"""
