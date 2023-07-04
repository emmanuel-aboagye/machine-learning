"""
Data preprocessing: it is assumed that most of the exploratory data analysis has been done
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, HeNormal
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope

# set random seed for reproducability
SEED=0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.experimental.numpy.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Import data using pandas
url = "link to your data"

# function to import data from given excel
def import_data(url):
  DataforML = pd.read_excel(url)
  DataforML.set_index("Name", inplace=True) # use this if needed else comment this part
  return DataforML

# function for replacing missing data 
def replace_missing_data(data):
  imputer = KNNImputer(n_neighbors = 5)
  data_ = imputer.fit_transform(data)
  data_ = pd.DataFrame(data_, columns=data.columns, index=data.index)
  return data_

# function to select the top 5 features based on R2 score for model development
def feat_selection(X,y,n_feat_to_sel=5):
  Xtrain_fs, Xtest_fs, ytrain_fs, ytest_fs = train_test_split(X,y, test_size=0.3, random_state=42)
  LR = LinearRegression()
  SFS = SequentialFeatureSelector(LR, n_features_to_select=n_feat_to_sel, direction='backward', scoring=None, cv=5)
  sfs = SFS.fit(Xtrain_fs, ytrain_fs)
  selfea = Xtrain_fs.columns[sfs.get_support()]
  return selfea

# function to split scaled data into training, cross-validation (or development) and testing sets
def data_split(X, y):
  X_train ,X_splt ,y_train ,y_splt = train_test_split(X,y,test_size=0.3,random_state=42)
  X_cv ,X_test ,y_cv ,y_test = train_test_split(X_splt,y_splt,test_size=0.5,random_state=42)
  return [X_train, X_cv, X_test, y_train, y_cv, y_test]

# Import data 
Data    = import_data(url)
X_data = Data.iloc[:,:-1]
y_data  = Data.iloc[:,-1]
y_data  = replace_missing_data(y_data) # assuming there is missing data in labels, replace them 

# select the top 5 important features from data
Xsel_data = feat_selection(X_data, y_data, n_feat_to_sel=5)

# combine the selected features (5) with the label
sel_data = pd.concat([X_data[Xsel_data], y_data, axis=1)

# split the data 
Xtrain, Xcv, Xtest, ytrain, ycv, ytest = data_split(sel_data_CC.iloc[:,:-1], sel_data_CC.iloc[:,-1])

# define parameter search space for each parameter of the nueral network
params = {'learning_rate'       : hp.uniform('learning_rate', 0.001,0.3),                    # learning rate parameter
          'units'               : scope.int(hp.quniform('units',10,250,10)),                 # number of neurons for hidden layers
          'units1'              : scope.int(hp.quniform('units1',10,250,10)),                # number of neurons for input layer
          'batch_size'          : scope.int(hp.quniform('batch_size',0,250,25)),             # batch size
          'layers'              : scope.int(hp.quniform('layers',1,5,1)),                    # number of layers
          'activation'          : hp.choice('activation',['relu','tanh','elu', 'sigmoid', 'leaky_relu']),         # activation function options for hidden layer. add/remove as needed
          'alpha'               : hp.uniform('alpha', 0.001,0.3),                                                 # hidden layer learning rate for LeakyReLU activation function 
          'activation1'         : hp.choice('activation1',['relu','tanh','elu', 'sigmoid', 'leaky_relu']),        # activation function options for input layer. add/remove as needed
          'alpha1'              : hp.uniform('alpha1', 0.001,0.3),                                                # input layer learning rate for LeakyReLU activation function
          #'activation2'         : hp.choice('activation2',['relu','tanh','elu', 'sigmoid']),                     # can be used for the output layer activation function
          'rate'                : hp.uniform('rate',0.001,0.5),                                                   # dropout rate for hidden layer
          'rate1'               : hp.uniform('rate1',0.001,0.5),                                                  # dropout rate for input layer
          'epochs'              : scope.int(hp.choice('epochs', range(500,5000,500))),                            # number of epochs
          'loss'                : hp.choice('loss', ['mean_squared_error', 'mean_absolute_error', 'logcosh', 'mean_squared_logarithmic_error']),    # loss function. add/remove as needed
          'initializer'         : hp.choice('initializer',[GlorotUniform(seed=0), GlorotNormal(seed=0), HeNormal(seed=0)]),                         # hidden layer weight initialization. add/remove as needed
          'initializer1'        : hp.choice('initializer1',[GlorotUniform(seed=0), GlorotNormal(seed=0), HeNormal(seed=0)]),                        # input layer weight initialization. add/remove as needed
          # 'regularizer': hp.choice('regularizer', [l2(i) for i in np.logspace(-4, -1, 4)]), for only l2 regularization               # can be used for regularization
        }

# define model for hyperopt
def nn_model_hpt(params):
  tf.keras.utils.set_random_seed(SEED)
  model = Sequential()
  
  # input layer
  if  params['activation1'] == 'leaky_relu': 
      model.add(Dense(units=params['units1'], input_dim=Xtrain.shape[1], kernel_initializer=params['initializer1']))
      model.add(LeakyReLU(alpha=params['alpha1']))
  else:
      model.add(Dense(units=params['units1'], input_dim=Xtrain.shape[1], activation=params['activation1'], kernel_initializer=params['initializer1']))
  model.add(Dropout(rate=params['rate1'])) 
  
  # hidden layers
  for i in range(params['layers']):
      if  params['activation'] == 'leaky_relu': 
          model.add(Dense(units=params['units'], kernel_initializer=params['initializer']))
          model.add(LeakyReLU(alpha=params['alpha']))
      else:
          model.add(Dense(units=params['units'], activation=params['activation'], kernel_initializer=params['initializer']))
      model.add(Dropout(rate=params['rate']))
  
  # output layer (depending on the expected output, you can change the activation or include that in the search space
  model.add(Dense(units=1, activation='relu', kernel_initializer=params['initializer1']))

  # compile model
  model.compile(loss=params['loss'], optimizer=Adam(learning_rate=params['learning_rate']))

  # fit model
  model.fit(Xtrain,
            ytrain,
            epochs=params['epochs'], 
            batch_size=params['batch_size'],
            shuffle=True,                             # becareful not to shuffle time-series data
            verbose=0,
            validation_data=(Xcv,ycv),
            )
  # y_pred_tr = model.predict(Xtrain, verbose=0)
  y_pred_cv = model.predict(Xcv, verbose=0)
  score = mean_squared_error(ycv, y_pred_cv)
  return {'loss': score, 'status': STATUS_OK}

# run hyperopt
trials = Trials()
Best = fmin(fn=nn_model_hpt, 
            space=params, 
            algo=tpe.suggest, 
            max_evals=2000,
            trials=trials)

# save selected features and the best hyperparameters in a text file
with open("hppt_nn_model.txt", "a") as file:
    file.writelines("Selected features: " + str(Xsel_data))
    file.write("\n------------------------------------------------\n")
    file.writelines("Best hyperparameters: " + str(space_eval(params,Best)))
    file.write("\n------------------------------------------------\n")
file.close()
  
"""
If you are constrained by data, one of the ways I found to improve my model was to try and minimize the summantion of the 
mean-squared-error for both the training and validation set [. Ideally, you would want to assign weights to each based on 
importance (normally importance is given to the validation mean-squared-error). This idea is similar to assigning weights 
when building a physics-informed neural network
"""  
