import preprocess as prp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import preprocess as prp
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump, load
import numpy as np
from sklearn.model_selection import GridSearchCV
from pactools.grid_search import GridSearchCVProgressBar


#pkr_data
#TODO move into preprocess
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.target = 'hand'
pkr_data.features = pkr_data.all.columns[pkr_data.all.columns != pkr_data.target]

pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data()
# ab_data.encode = ['room_type']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)

ab_train_scores = {}
ab_test_scores = {}
pkr_train_scores = {}
pkr_test_scores = {}
pkr_test_auc = {}
pkr_train_auc = {}


#PK model
loaded_clf_pk = load('clf_NN_pkr data_final_0.9728.joblib')
clf_pk = loaded_clf_pk.named_steps.gridsearchcv.best_estimator_

#AB model
loaded_clf_ab = load('clf_NN_abnb_final_0.8273.joblib')
clf_ab = loaded_clf_ab.named_steps.gridsearchcv.best_estimator_


print("Models Initiated")


for n in range(1,10):
# for wt in np.linspace(0,1,20):
    hidden_layers = [n for k in range(n)]
    hidden_layers.insert(0,len(ab_data.features))
    hidden_layers.append(len(ab_data.all[ab_data.target].unique()))
    hidden_layers = tuple(hidden_layers)
    NN_ab = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layers, random_state=1,max_iter=5000)
    # NN_ab = MLPClassifier(solver='lbfgs', learning_rate_init=n,hidden_layer_sizes=(len(ab_data.features),n,n, len(ab_data.all[ab_data.target].unique())), random_state=1,max_iter=1000)

    NN_ab.fit(ab_data.x_train,ab_data.y_train)
    ab_train_score = NN_ab.score(ab_data.x_train,ab_data.y_train)
    ab_test_score = NN_ab.score(ab_data.x_test,ab_data.y_test)
    ab_train_scores[n] = ab_train_score
    ab_test_scores[n] = ab_test_score

# for wt in np.linspace(0,1,20):
    hidden_layers_pk = [n for k in range(n)]
    hidden_layers_pk.insert(0,len(pkr_data.features))
    hidden_layers_pk.append(len(np.unique(pkr_data.all[pkr_data.target])))
    hidden_layers_pk = tuple(hidden_layers_pk)

    NN_pkr = MLPClassifier(solver='lbfgs', learning_rate_init=n,hidden_layer_sizes=hidden_layers_pk, random_state=1,max_iter=1000)

    NN_pkr.fit(pkr_data.x_train,pkr_data.y_train)
    pkr_train_score = NN_pkr.score(pkr_data.x_train,pkr_data.y_train)
    pkr_test_score = NN_pkr.score(pkr_data.x_test,pkr_data.y_test)
    pkr_train_scores[n] = pkr_train_score
    pkr_test_scores[n] = pkr_test_score
    pkr_test_auc[n] = metrics.roc_auc_score(pkr_data.y_test, NN_pkr.predict(pkr_data.x_test), average='macro')
    pkr_train_auc[n] = metrics.roc_auc_score(pkr_data.y_train, NN_pkr.predict(pkr_data.x_train), average='macro')

marker = 1

plt.plot(list(ab_train_scores.keys()),list(ab_train_scores.values()),'o-',label='ab_training_accuracy', color="b")
plt.plot(list(ab_test_scores.keys()),list(ab_test_scores.values()),'x-',label='ab_test_accuracy', color="b")
plt.plot(list(pkr_train_scores.keys()),list(pkr_train_scores.values()),'o-',label='pkr_training_accuracy', color="r")
plt.plot(list(pkr_test_scores.keys()),list(pkr_test_scores.values()),'x-',label='pkr_test_accuracy', color="r")


# plt.plot(list(pkr_train_auc.keys()),list(pkr_train_auc.values()),'x--',label='pkr_training_roc', color="r")
# plt.plot(list(pkr_test_auc.keys()),list(pkr_test_auc.values()),'x--',label='pkr_test_roc', color="b")


plt.xlabel('Hidden Layer size (nxn)')
# plt.xlabel('Weight of Rare Hand')

plt.ylabel('Accuracy (%)')
plt.title('NN Accuracy vs Hidden Layer Size')
plt.legend(loc='best')
# plt.xticks(list(ab_train_scores.keys()))
# plt.ylim(0,1.0)
# plt.show()
plt.savefig('NN_hiddenlayer_var')
plt.close()
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
#Gini Vs Entropy
# clf_ab = DecisionTreeClassifier(random_state=0,criterion='gini',splitter='best')


#
# ab_train_scores = [clf.score(ab_data.X_train, ab_data.y_train) for clf in clfs]
# ab_test_scores = [clf.score(ab_data.X_test, ab_data.y_test) for clf in clfs]
#
# path = clf.cost_complexity_pruning_path(ab_data.X_train, ab_data.y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities


# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train",
#         drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test",
#         drawstyle="steps-post")
# ax.legend()
# plt.show()
#
#
