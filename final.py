"""
File Name     : final.py
Purpose       : Predict Tropical Cyclone Intensity Change
Authors       : Eric Wu, Vania Revelina, Peng Zhang, Vincent Chen
"""

# Import all necessary packages
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.api as sm
from sklearn.externals import joblib

# Create necessary functions
print("Defining necessary functions...")
def open_all_prcp(startyear, endyear):
  """Opens all prcp dataset and saves them into 1 list of dataframes"""
  filenames=[]
  file_open=[]
  prcpdf=[]
  dflist=[]
  for i in range(startyear,endyear+1):
    fname='Prcp'+str(i)+'.nc'
    filenames.append(fname)
  for file in filenames:
    file_open.append(xr.open_dataset(file))
  for i in range(startyear,endyear+1):
    name='prcp'+str(i)
    prcpdf.append(name)
  for prcp,i in zip(file_open,range(endyear-startyear+1)):
    exec(f'{prcpdf[i]}=prcp.to_dataframe()')
    exec(f'dflist.append({prcpdf[i]})')
  return dflist

def open_all_testprcp(startyear, endyear):
  """Opens all prcp dataset and saves them into 1 list of dataframes"""
  filenames=[]
  file_open=[]
  prcpdf=[]
  dflist=[]
  for i in range(startyear,endyear+1):
    fname='test_Prcp'+str(i)+'.nc'
    filenames.append(fname)
  for file in filenames:
    file_open.append(xr.open_dataset(file))
  for i in range(startyear,endyear+1):
    name='prcp'+str(i)
    prcpdf.append(name)
  for prcp,i in zip(file_open,range(endyear-startyear+1)):
    exec(f'{prcpdf[i]}=prcp.to_dataframe()')
    exec(f'dflist.append({prcpdf[i]})')
  return dflist

def clean_prcp(dflist):
  """Cleans all prcp datasets and merge them into 1 dataframe"""
  prcpall = pd.concat(dflist)
  prcpall.replace(-999.0, np.nan, inplace=True)
  prcpall.dropna(subset=['dvmax','clat','clon'],inplace=True)
  prcpall.reset_index(level=['Date','RHOUR','RLAT','RLON'],inplace=True)
  prcpall['Date'] = pd.to_datetime(prcpall['Date'],format='%Y%m%d%H')
  prcpall.rename(columns={'Date':'DATE'},inplace=True)
  prcpall.drop(['RHOUR','RLAT','RLON'],axis=1,inplace=True)
  prcp_group=prcpall.groupby(['DATE','id']).mean().reset_index(level=['DATE','id'])
  prcp_group.dropna(inplace=True)
  return prcp_group

def clean_testprcp(dflist):
  """Cleans all prcp datasets and merge them into 1 dataframe"""
  prcpall = pd.concat(dflist)
  prcpall.replace(-999.0, np.nan, inplace=True)
  prcpall.dropna(subset=['clat','clon'],inplace=True)
  prcpall.reset_index(level=['Date','RHOUR','RLAT','RLON'],inplace=True)
  prcpall['Date'] = pd.to_datetime(prcpall['Date'],format='%Y%m%d%H')
  prcpall.rename(columns={'Date':'DATE'},inplace=True)
  prcpall.drop(['RHOUR','RLAT','RLON'],axis=1,inplace=True)
  prcp_group=prcpall.groupby(['DATE','id']).mean().reset_index(level=['DATE','id'])
  prcp_group.dropna(inplace=True)
  return prcp_group

def mergeprcpships(clprcp, ships):
  """Merges the combined prcp datasets and the ships dataset"""
  datas = pd.merge(clprcp, ships, left_on='DATE', right_on='DATE', how='inner')
  datas.dropna(inplace=True)
  datas.drop(['id_y','dvmax_x','clat_x','clon_x'],axis=1,inplace=True)
  datas.drop_duplicates(inplace=True)
  return datas

def mergetestprcpships(clprcp, ships):
  """Merges the combined prcp datasets and the ships dataset"""
  datas = pd.merge(clprcp, ships, left_on='DATE', right_on='DATE', how='inner')
  datas.dropna(inplace=True)
  datas.drop(['id_y','clat_x','clon_x'],axis=1,inplace=True)
  datas.drop_duplicates(inplace=True)
  return datas

def POD(ytest, ypred):
  """Function to calculate POD
     input is a confusion matrix
     outputs the POD score"""
  confmat = confusion_matrix(ytest, ypred)
  return confmat[0,0]/(confmat[0,0]+confmat[0,1])

def FAR(ytest, ypred):
  """Function to calculate FAR
     input is a confusion matrix
     outputs the FAR score"""
  confmat = confusion_matrix(ytest, ypred)
  return confmat[1,0]/(confmat[0,0]+confmat[1,0])

def PSS(ytest, ypred):
  """Function to calculate PSS
     input is a confusion matrix
     outputs the PSS score"""
  confmat = confusion_matrix(ytest, ypred)
  return ((confmat[0,0]*confmat[1,1])-(confmat[1,0]*confmat[0,1]))/((confmat[0,0]+confmat[0,1])*(confmat[1,0]+confmat[1,1]))

def print_scores(modelname, ytest, ypred):
  """Prints the POD, FAR, and PSS scores of a model
     input is the model name, test labels and predicted labels"""
  print("For the {} Model:\n".format(modelname),
        "POD is {}\n".format(POD(ytest, ypred)),
        "FAR is {}\n".format(FAR(ytest, ypred)),
        "PSS is {}.".format(PSS(ytest, ypred)))

def load_data(datapathnc):
  """Opens a .nc extension file and returns a dataframe"""
  shipdf = xr.open_dataset(datapathnc).to_dataframe()
  shipdf.index = pd.to_datetime(shipdf.index,format='%Y%m%d%H')
  shipdf.reset_index(inplace=True)
  cols_keep=['DATE','dvmax', 'clat', 'clon', 'id','MSLP','PER','SHRD','NOHC','SHRG','TPWC','TADV','PSLV']
  shipdf=shipdf[cols_keep]
  shipdf.dropna(inplace=True)
  return shipdf

def load_testdata(datapathnc):
  """Opens a .nc extension file and returns a dataframe"""
  shipdf = xr.open_dataset(datapathnc).to_dataframe()
  shipdf.index = pd.to_datetime(shipdf.index,format='%Y%m%d%H')
  shipdf.reset_index(inplace=True)
  cols_keep=['DATE','clat', 'clon', 'id','MSLP','PER','SHRD','NOHC','SHRG','TPWC','TADV','PSLV']
  shipdf=shipdf[cols_keep]
  shipdf.dropna(inplace=True)
  return shipdf

def ri(x):
  """Returns RI classification value element-wise
     input is the dvmax value
     For use with the apply function on dataframe"""
  if x>=25:
      return 1
  else:
      return 0

# load and clean all training data
print("LOADING ALL TRAINING DATA")
print("Loading SHIPS_1998-2010.nc...")
shipdf = load_data("SHIPS_1998-2010.nc")
print("Loading and cleaning Prcpxxxx.nc data from 1998-2004. This may take a while...")
prcp_gb1 = clean_prcp(open_all_prcp(1998,2004))
print("Loading and cleaning Prcpxxxx.nc data from 2005-2010. This may take a while...")
prcp_gb2 = clean_prcp(open_all_prcp(2005,2010))
print("Merging all training data...")
half_data1 = mergeprcpships(prcp_gb1, shipdf)
half_data2 = mergeprcpships(prcp_gb2, shipdf)
all_data = pd.concat([half_data1, half_data2])

print("Generating RI variable...")
# create RI variable
all_data['RI']=all_data['dvmax_y'].apply(ri)

# split the data into dependent and independent variables
y=all_data['RI']
X=all_data.drop(['DATE','RI','id_x','dvmax_y'],axis=1)

# split data into training and validation set
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

print("Oversampling to handle imbalanced data...")
# Oversample imbalanced data by SMOTE
smote = SMOTE(ratio='minority')
X_train, y_train = smote.fit_sample(X_tr, y_tr)

# Logistic Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate the model
log_reg_ = LogisticRegression()
# Create hyperparameter grid
param_grid = [{'penalty': ['l1'], 'solver': [ 'liblinear', 'saga']}, {'penalty': ['l2'], 'solver': ['sag','lbfgs','newton-cg']}]
# Instantiate the GridSearch
log_cv = GridSearchCV(log_reg_, param_grid, cv=5)
# Fit the model
log_cv.fit(X_train, y_train)
# View best hyperparameters
print("Tuned Logistic Regression Parameter: {}".format(log_cv.best_params_))
# View best score
print("Tuned Logistic Regression Accuracy: {}".format(log_cv.best_score_))
# Instantiate the model with best parameters
log_reg = LogisticRegression(C=1.62, penalty='l1')
# Fit the model
log_reg.fit(X_train, y_train)
# Predict the labels of the validation set: y_pred1
y_pred1 = log_reg.predict(X_test)
print('Logistic Regression Confusion Matrix\n', confusion_matrix(y_test, y_pred1))
print_scores('Logistic Regression', y_test, y_pred1)


# Decision Tree Classification Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate the model
tree1 = DecisionTreeClassifier()
# Fit the model
tree1.fit(X_train, y_train)
# Predict the labels of the validation set: y_pred2
y_pred2 = tree1.predict(X_test)
print('DecisionTreeClassifier 1 Confusion Matrix\n', confusion_matrix(y_test, y_pred2))
print_scores('DecisionTreeClassifier 1', y_test, y_pred2)


# Decision Tree Classification Model 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate the model
tree2_ = DecisionTreeClassifier()
# Create parameters to test
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
# Instantiate the RandomSearch
tree_cv = RandomizedSearchCV(tree2_, param_dist, cv=5)
# Fit the model
tree_cv.fit(X_train, y_train)
# View best hyperparameters
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
# View best score
print("Best score is {}".format(tree_cv.best_score_))
# Instantiate the model with best parameters
tree2 = DecisionTreeClassifier(criterion= 'entropy', max_depth= None, max_features= 7, min_samples_leaf= 5)
# Fit the model
tree2.fit(X_train, y_train)
# Predict the labels of the validation set: y_pred3
y_pred3 = tree2.predict(X_test)
print('DecisionTreeClassifier 2 Confusion Matrix\n', confusion_matrix(y_test, y_pred3))
print_scores('DecisionTreeClassifier 2', y_test, y_pred3)


# Decision Tree Classification Model 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using StandardScaler()
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
# Instantiate the model
tree3 = DecisionTreeClassifier(criterion= 'entropy', max_depth= None, max_features= 6, min_samples_leaf= 6)
# Fit the model
tree3.fit(X_train_sc, y_train)
# Predict the labels of the validation set: y_pred4
y_pred4 = tree3.predict(X_test_sc)
print('DecisionTreeClassifier 3 Confusion Matrix\n', confusion_matrix(y_test, y_pred4))
print_scores('DecisionTreeClassifier 3', y_test, y_pred4)


# SVC Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
# Instantiate the GridSearchCV
svc_cv = GridSearchCV(pipeline, param_grid=parameters)
# Fit the model
svc_cv.fit(X_train,y_train)
# Predict the labels of the test set: y_pred5
y_pred5 = svc_cv.predict(X_test)
print('SVC Model Confusion Matrix\n', confusion_matrix(y_test, y_pred5))
print_scores('SVC Model', y_test, y_pred5)


# Random Forest Model 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate the model
forest1 = RandomForestClassifier()
# Fit the model
forest1.fit(X_train, y_train)
# Predict the labels of the test set: y_pred6
y_pred6 = forest1.predict(X_test)
print('RandomForestClassifier 1 Confusion Matrix\n', confusion_matrix(y_test, y_pred6))
print_scores('RandomForestClassifier 1', y_test, y_pred6)


# Random Forest Model 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate the model
forest2 = RandomForestClassifier(n_estimators=100, criterion="entropy")
# Fit the model
forest2.fit(X_train, y_train)
# Predict the labels of the test set: y_pred6
y_pred7 = forest2.predict(X_test)
print('RandomForestClassifier 2 Confusion Matrix\n', confusion_matrix(y_test, y_pred7))
print_scores('RandomForestClassifier 2', y_test, y_pred7)

# Save the model for further use
filename = 'finalized_model.sav'
joblib.dump(forest2, filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process the test data
print("Loading test_SHIPS_2011-2014.nc...")
shipdf = load_testdata("test_SHIPS_2011-2014.nc")
print("Loading and cleaning Prcpxxxx.nc data from 2011-2014. This may take a while...")
prcp_gb1 = clean_testprcp(open_all_testprcp(2011,2014))
X_realtest = mergetestprcpships(prcp_gb1, shipdf)

# Test the Decision Tree Model 2 with the Real Test Data 2011-2014
Xfinal=X_realtest.drop(['DATE','id_x'],axis=1)
# Load the model
forest_model = joblib.load('finalized_model.sav')
print("Predicting RI values for 2011-2014...")
y_pred_new = forest_model.predict(Xfinal)
X_realtest['RI']=y_pred_new


results=X_realtest.loc[:,['DATE','clat_y','clon_y','RI']]
results.rename(columns={'DATE':'Date','clat_y':'clat','clon_y':'clon','RI':'predicted_RI'},inplace=True)
results.drop_duplicates(subset='Date', inplace=True)
results.to_csv(path_or_buf='solutions_LionsBluffTeam.csv',index=False)
print("All done! A csv file named [solutions_LionsBluffTeam.csv] has been created in the same folder containing the prediction results for years 2011-2014.")
