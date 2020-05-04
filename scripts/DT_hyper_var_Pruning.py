import preprocess as prp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import preprocess as prp
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
print("Models Initiated")



ab_train_scores = {}
ab_test_scores = {}
pkr_train_scores = {}
pkr_test_scores = {}
pkr_test_auc = {}
pkr_train_auc = {}
crit = 'gini'
split_crit = 'best'

#TODO test gini vs entropy vs bes vs random
for alpha in np.linspace(0,0.005,40):
# for wt in np.linspace(0,1,20):
    DT_ab = DecisionTreeClassifier(ccp_alpha=alpha)

    DT_ab.fit(ab_data.x_train,ab_data.y_train)
    ab_train_score = DT_ab.score(ab_data.x_train,ab_data.y_train)
    ab_test_score = DT_ab.score(ab_data.x_test,ab_data.y_test)
    ab_train_scores[alpha] = ab_train_score
    ab_test_scores[alpha] = ab_test_score

    DT_pkr = DecisionTreeClassifier(ccp_alpha=alpha)
    DT_pkr.fit(pkr_data.x_train,pkr_data.y_train)
    pkr_train_score = DT_pkr.score(pkr_data.x_train,pkr_data.y_train)
    pkr_test_score = DT_pkr.score(pkr_data.x_test,pkr_data.y_test)
    pkr_train_scores[alpha] = pkr_train_score
    pkr_test_scores[alpha] = pkr_test_score

marker = 1

plt.plot(list(ab_train_scores.keys()),list(ab_train_scores.values()),'o-',label='ab_training_accuracy', color="b")
plt.plot(list(ab_test_scores.keys()),list(ab_test_scores.values()),'x-',label='ab_test_accuracy', color="b")
plt.plot(list(pkr_train_scores.keys()),list(pkr_train_scores.values()),'o-',label='pkr_training_accuracy', color="r")
plt.plot(list(pkr_test_scores.keys()),list(pkr_test_scores.values()),'x-',label='pkr_test_accuracy', color="R")



plt.xlabel('Cost Complexity Pruning Alpha')

plt.ylabel('Accuracy (%)')
plt.title('DT Accuracy vs Cost Complexity Alpha Pruning')
plt.legend(loc='best')
# plt.xticks(list(ab_train_scores.keys()))
# plt.ylim(0,1.0)
# plt.show()
plt.savefig('DT_alpha_var_pruning')
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
