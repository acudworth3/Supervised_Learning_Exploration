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


#pkr_data
#TODO move into preprocess
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data()
# ab_data.encode = ['room_type']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)




#validate
clf_a = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(len(ab_data.features),5, len(ab_data.all[ab_data.target].unique())), random_state=1)
clf_a.fit(ab_data.x_train,ab_data.y_train)
ab_data.y_predict=clf_a.predict(ab_data.x_test)
score = metrics.accuracy_score(ab_data.y_test,ab_data.y_predict)
y_prob = clf_a.predict_proba(ab_data.x_test)
roc_score = metrics.roc_auc_score(pkr_data.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data

print("ab_data score:",score)
#
#
clf_p = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(len(pkr_data.features),5,10), random_state=1)
clf_p.fit(pkr_data.x_train,pkr_data.y_train)
pkr_data.y_predict=clf_p.predict(pkr_data.x_test)
score = metrics.accuracy_score(pkr_data.y_test,pkr_data.y_predict)
# score2 = metrics.roc_auc_ovr(pkr_data.y_test,pkr_data.y_predict)

print("pk_data score:",score)

#TRAIN
# #ab Data
ab_r = range(1,3)
#
# #create structure features:l_1:l_2:target
ab_hidden_layers = [(len(ab_data.features),l_1,l_2, len(ab_data.all[ab_data.target].unique())) for l_1 in ab_r for l_2 in ab_r]
ab_hidden_layers.append((len(ab_data.features),len(ab_data.all[ab_data.target].unique())))
parameters_a = {'solver':['lbfgs', 'sgd', 'adam'], 'alpha':list(10.0 ** -np.arange(1, 7)),'hidden_layer_sizes':ab_hidden_layers}
# parameters_a = {'solver':['lbfgs']}


pk_r = range(1,2)
#
# #create structure features:l_1:l_2:target
pk_hidden_layers = [(len(pkr_data.features),l_1,l_2, 10) for l_1 in pk_r for l_2 in pk_r]
pk_hidden_layers.append((len(pkr_data.features),10))
parameters_pk = {'solver':['lbfgs', 'sgd', 'adam'], 'alpha':list(10.0 ** -np.arange(1, 7)),'hidden_layer_sizes':pk_hidden_layers}
# parameters_pk = {'solver':['lbfgs']}

#
# #
def NN_accr_gridSearch(data_obj, params):
    # TODO add progress bar

    clf_grid = make_pipeline(preprocessing.StandardScaler(),
                             GridSearchCV(MLPClassifier(),
                                          param_grid=params,
                                          cv=5,
                                          refit=True, n_jobs=3, verbose=2))

    # clf_grid = make_pipeline(preprocessing.StandardScaler(),
    #                          GridSearchCV(AdaBoostClassifier(),
    #                                       param_grid=params,
    #                                       cv=5,
    #                                       refit=True, n_jobs=3, verbose=2,scoring='roc_auc_ovr'))


    clf_grid.fit(data_obj.x_train, data_obj.y_train)

    y_prob = clf_grid.predict_proba(data_obj.x_test)
    y_predict = clf_grid.predict(data_obj.x_test)

    score = clf_grid.score(data_obj.x_test, data_obj.y_test)
    # score = metrics.roc_auc_score(data_obj.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
    # score = metrics.roc_auc_score(data_obj.y_test, y_predict, multi_class='ovr', average='macro',max_fpr=1.0)  # for pkr data


    print("NN grid search:", score)
    dump(clf_grid, 'clf_NN_' + str(data_obj.title) +'_final_'+ str(round(score, 4)) + '.joblib')



def NN_roc_gridSearch(data_obj, params):
    # TODO add progress bar

    clf_grid = make_pipeline(preprocessing.StandardScaler(),
                             GridSearchCV(MLPClassifier(),
                                          param_grid=params,scoring='roc_auc_ovr',
                                          cv=5,
                                          refit=True, n_jobs=3, verbose=2))

    # clf_grid = make_pipeline(preprocessing.StandardScaler(),
    #                          GridSearchCV(AdaBoostClassifier(),
    #                                       param_grid=params,
    #                                       cv=5,
    #                                       refit=True, n_jobs=3, verbose=2,scoring='roc_auc_ovr'))


    clf_grid.fit(data_obj.x _train, data_obj.y_train)

    # y_prob = clf_grid.predict_proba(data_obj.x_test)
    # y_predict = clf_grid.predict(data_obj.x_test)

    # score = clf_grid.score(data_obj.x_test, data_obj.y_test)
    # score = metrics.roc_auc_score(data_obj.y_test,y_predict,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
    # score = metrics.roc_auc_score(data_obj.y_test, y_predict, multi_class='ovr', average='macro',max_fpr=1.0)  # for pkr data


    print("NN roc grid search:", 000)
    dump(clf_grid, 'clf_NN_' + str(data_obj.title) +'_final_roc00000.joblib')


NN_accr_gridSearch(pkr_data,parameters_pk)
NN_roc_gridSearch(pkr_data,parameters_pk)
NN_roc_gridSearch(ab_data,parameters_a)

NN_accr_gridSearch(ab_data,parameters_a)
