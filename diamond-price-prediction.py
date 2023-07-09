"""
@author: aboag
"""
#%% import packages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

#%% lets define some functions
# function to import data
def import_data(path):
    data = pd.read_csv(path)
    data = data.drop('Unnamed: 0', axis=1)
    df = data.copy()
    return df

# function to select columns with datatypes 'object'
def sect_dtypes_object_col(df):
    categorical_data = []
    for i in df.select_dtypes(include='object').columns:       
        categorical_data.append(i)
    return categorical_data

# function to perform one-hot encoding on datatypes 'object'
def one_hot_encoder(df, cat_columns):
    transformer = make_column_transformer(
                (OneHotEncoder(), cat_columns),
                remainder= 'passthrough',        # specifies that all other columns should be left untouched
        )   
    transformed = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    return transformed_df

# function to rename columns
def rename_cols(df):
    colmns_ = [] 
    colmns = []
    acutal_col = []
    for i in df:
        colmns_.append(i)
    for j in colmns_:
        colmns.append(j.split('__'))
    for k in range(len(colmns)):
        acutal_col.append(colmns[k][1])   
    column_dict = dict(zip(colmns_,acutal_col))
    transformed_df = df.rename(columns=column_dict)
    return transformed_df

# function to remove outliers (values with z-score > 3)
def outlier_removal(df):
    z1 = np.abs(stats.zscore(df.y))
    print(len(np.where(z1>3)[0]))
    transformed_df = df[(z1<3)]

    # for column 'x'
    z2 = np.abs(stats.zscore(df.x))
    print(len(np.where(z2>3)[0]))
    transformed_df = df[(z2<3)]

    # for column 'z'
    z3 = np.abs(stats.zscore(df.z))
    print(len(np.where(z3>3)[0]))
    transformed_df = df[(z3<3)]

    # for column 'table'
    z4 = np.abs(stats.zscore(df.table))
    print(len(np.where(z4>3)[0]))
    transformed_df = df[(z4<3)]
    return transformed_df

# function to split data
def split_data(feature, label, split=0.2, rs=42, val_split=False):
    if val_split:
        X_train, X_test, y_train, y_test = train_test_split(feature,label,test_size=split,random_state=rs)
        X_val = [None] * len(X_test)
        y_val = [None] * len(y_test)
    else:
        X_train, X_, y_train, y_ = train_test_split(feature,label,test_size=split,random_state=rs)
        X_val, X_test, y_val, y_test = train_test_split(X_,y_,test_size=0.5,random_state=rs)
    return [X_train, X_val, X_test, y_train, y_val, y_test]

# pipeline function
def build_pipeline():
    pipelines = {
        'lin_reg': make_pipeline(StandardScaler(), LinearRegression()),
        'rft_reg': make_pipeline(StandardScaler(), RandomForestRegressor()),
        'xgb_reg': make_pipeline(StandardScaler(), XGBRegressor())
    }
    return pipelines

# function to fit model
def fit_models(pipelines, feature_train_set, label_train_set):
    fitted_model = {}
    for models_, pipelines_ in pipelines.items():
        model = pipelines_.fit(feature_train_set, label_train_set)
        fitted_model[models_] = model
    return fitted_model

# function to evaluate trained model
def evaluate_trained_model(fitted_model, feature_test_set, label_test_set):
    rmse = []
    r_squared = []
    model_s = []
    for i, models_ in fitted_model.items():
        label_pred = models_.predict(feature_test_set)
        model_s.append(i)
        rmse.append(np.sqrt(mean_squared_error(label_test_set, label_pred)))
        r_squared.append(r2_score(label_test_set, label_pred))
    
    rmse_results = zip(model_s,rmse)
    rsq_results  = zip(model_s,r_squared)
    print('root-mean-squared-error')
    for j in rmse_results: print(j)
    print('--------------------------------')
    print('r-squared value')
    for k in rsq_results: print(k)
    
    return [rmse, r_squared, model_s]

# function to make predictions
def make_predctions(fitted_model, feature_train_set, feature_test_set):
    label_train_pred = []
    label_test_pred  = []
    model_ = []
    for i, model in fitted_model.items():
        train_pred = model.predict(feature_train_set)
        test_pred  = model.predict(feature_test_set)
        label_train_pred.append(train_pred)
        label_test_pred.append(test_pred)
        model_.append(i)
    return [label_train_pred, label_test_pred, model_]

# plot results
def plot_results(model_fitted, rmse, r_squared, label_train, label_train_pred , label_test, label_test_pred , plot_type='bar'):
    if plot_type == 'bar':
        plt.figure(figsize=(7,7), dpi=700)
        plt.subplot(2,1,1)
        sns.barplot(x=model_fitted, y=rmse)
        plt.xlabel('Machine Learning Model')
        plt.ylabel('root-mean-squared-error')
        
        plt.subplot(2,1,2)
        sns.barplot(x=model_fitted, y=r_squared)
        plt.xlabel('Machine Learning Model')
        plt.ylabel('r-squared value')
        plt.show()
    elif plot_type == 'scatter':
        Y1_train_pred = label_train_pred[0]
        Y2_train_pred = label_train_pred[1]
        Y3_train_pred = label_train_pred[2]
        
        Y1_test_pred = label_test_pred[0]
        Y2_test_pred = label_test_pred[1]
        Y3_test_pred = label_test_pred[2]
        
        plt.figure(figsize=(12,12), dpi=700)
        plt.subplots_adjust(hspace=0.35)
        plt.subplot(3,1,1)
        plt.plot(label_train, Y1_train_pred, 'b.', label='train')
        plt.plot(label_test, Y1_test_pred, 'r.', label='test')
        plt.xlabel('True Price')
        plt.ylabel('Predicted Price')
        plt.legend(loc='best')
        plt.title(str(model_fitted[0]))
        
        plt.subplot(3,1,2)
        plt.plot(label_train, Y2_train_pred, 'b.', label='train')
        plt.plot(label_test, Y2_test_pred, 'r.', label='test')
        plt.xlabel('True Price')
        plt.ylabel('Predicted Price')
        plt.legend(loc='best')
        plt.title(str(model_fitted[1]))
        
        plt.subplot(3,1,3)
        plt.plot(label_train, Y3_train_pred, 'b.', label='train')
        plt.plot(label_test, Y3_test_pred, 'r.', label='test')
        plt.xlabel('True Price')
        plt.ylabel('Predicted Price')
        plt.legend(loc='best')
        plt.title(str(model_fitted[2]))
        plt.show()
    else:
        print("Error: incorrect input, plot_type has to be either 'bar' or 'scatter'")

#%% import data
path = "Diamonds Prices2022.csv"
df = import_data(path)

#%% Data preprocessing 
# perform one-hot encoding
res_df = one_hot_encoder(df, sect_dtypes_object_col(df))

# rename the cols after one-hot encoding
res_df = rename_cols(res_df)

# remove outliers
res_df = outlier_removal(res_df)

#%% split data into train-test
features_set = feature_set = res_df.drop(['price'], axis=1)
label_set = res_df['price']
feature_train, _ , feature_test, label_train, _ , label_test = split_data(features_set, label_set, split=0.2, rs=42, val_split=False)

#%% initialize pipeline
pipelines = build_pipeline()

#%% fit models
fitted_model = fit_models(pipelines, feature_train, label_train)
    
#%% evaluate the trained models
rmse, r_squared, model_fitted = evaluate_trained_model(fitted_model, feature_test, label_test)

#%% make prediction
Y_train_pred, Y_test_pred, model_name = make_predctions(fitted_model, feature_train, feature_test)

#%%
print(len(Y_test_pred[2]))
print(len(label_test))
#%% bar plot
plot_results(model_name, rmse, r_squared, label_train, Y_train_pred, label_test, Y_test_pred, plot_type='bar')

#%% scatter plot
plot_results(model_name, rmse, r_squared, label_train, Y_train_pred, label_test, Y_test_pred, plot_type='scatter')


#%% feature importance for random forest model
perm_results = permutation_importance(pipelines['rft_reg']['randomforestregressor'], 
                                      feature_test, 
                                      label_test,
                                      n_repeats=10, 
                                      random_state=42,
                                      )
perm_feature_importance_rf = pd.Series(perm_results.importances_mean, index=[i for i in features_set.columns])
fig, ax = plt.subplots()
perm_feature_importance_rf.plot.bar(yerr=perm_results.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
#%%

"""
From the results, random forest regression does better than both linear regression and 
xgboost regression

Linear regression is predicting negative values for the price, hence model doesn't make sense. Don't 
use linear regression. 

Feature importance plot indicates "caret" size has the highest influence on the price (not surprising!). Premium cut, G color, 
Clarities SI1, SI2, VS1, and VS2 are the only features that have positive permutation importance on the model 

Therefore, you can go ahead and develop a model for RandomForestRegressor using those features, and tune some hyperparameters of the model, if needed.
"""
