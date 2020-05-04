from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
import preprocess as prp
from sklearn import metrics
import numpy as np
from joblib import dump, load
from models import generic_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from seaborn import distplot
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV
from warnings import filterwarnings
filterwarnings('ignore')



#pkr_data
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])


#ab data
ab_data = prp.ab_data()
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]
ab_data.init_model_data(target=ab_data.target,features=ab_data.features)

#NN TRAIN
#ab Data
ab_r = range(1,4)
ab_hidden_layers = [(len(ab_data.features),l_1,l_2, len(ab_data.all[ab_data.target].unique())) for l_1 in ab_r for l_2 in ab_r]
ab_hidden_layers.append((len(ab_data.features),len(ab_data.all[ab_data.target].unique())))
parameters_a = {'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam'], 'alpha':list(10.0 ** -np.arange(1, 7)),'hidden_layer_sizes':ab_hidden_layers}

pk_r = range(1,4)
pk_hidden_layers = [(len(pkr_data.features),l_1,l_2, 10) for l_1 in pk_r for l_2 in pk_r]
pk_hidden_layers.append((len(pkr_data.features),10))
parameters_pk = {'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam'], 'alpha':list(10.0 ** -np.arange(1, 7)),'hidden_layer_sizes':pk_hidden_layers}

# clf_grid_a = make_pipeline(preprocessing.StandardScaler(),
#                     GridSearchCV(MLPClassifier(),
#                                  param_grid=parameters_a,scoring='roc_auc_ovr',
#                                  cv=5,
#                                  refit=True,n_jobs=2,verbose=10))
#
# clf_grid_a.fit(ab_data.x_train,ab_data.y_train)
# dump(clf_grid_a, 'clf_NN_grid_ab_dev.joblib')
#
#
# #pkr_data
# pk_r = range(1,4)
# pk_hidden_layers = [(len(pkr_data.features),l_1,l_2, 10) for l_1 in pk_r for l_2 in pk_r]
# pk_hidden_layers.append((len(pkr_data.features),10))
# parameters_pk = {'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam'], 'alpha':list(10.0 ** -np.arange(1, 7)),'hidden_layer_sizes':pk_hidden_layers}
# clf_grid_pk = make_pipeline(preprocessing.StandardScaler(),
#                     GridSearchCV(MLPClassifier(),
#                                  param_grid=parameters_pk,
#                                  cv=5,
#                                  refit=True,n_jobs=3,verbose=1))
#
#
# clf_grid_pk.fit(pkr_data.x_train,pkr_data.y_train)
# dump(clf_grid_pk, 'clf_NN_grid_pk_dev.joblib')




#TODO figure out what to do about label not in fold
#LOAD
ab_pass = prp.ab_data()
pkr_pass = prp.pkr_data()

# loaded_clf_ab = load('clf_NN_grid_ab_dev.joblib')
loaded_clf_pk = load('clf_NN_grid_pk_dev.joblib')
loaded_clf_ab = load('clf_SVM_gridabnb0.8236.joblib')


clf_pkr = loaded_clf_pk.named_steps.gridsearchcv.best_estimator_
clf_ab = loaded_clf_ab.named_steps.gridsearchcv.best_estimator_

pkr_accur_grid_model = generic_model(pkr_pass,clf_pkr,'clf_NN_grid_pk')

pkr_accur_grid_model.make_plots()
#ab_accur_grid_model.make_plots()

