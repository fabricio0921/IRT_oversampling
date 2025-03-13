import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SVMSMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids
from sklearn.metrics import accuracy_score, f1_score,  matthews_corrcoef,classification_report
import decodIRT_OtML as dIRT_OtML
import decodIRT_MLtIRT as dIRT_MLtIRT
import sys

def get_classication(resp,y):
    tmp = []
    for c,i in enumerate(y):
        if resp[c] == 1:
            tmp.append(i)
        elif i == 1:
            tmp.append(0)
        elif i == 0:
            tmp.append(1)
    return tmp

list_Datasets = ['pc4']

list_seeds = [10]
list_seeds_mtd = [10]

#list_over = ['adasyn','smote']
list_over = ['adasyn','smote','smoten','svmsmote','sem_nada']
#list_over = ['BorderlineSMOTE','RandomOverSampler','sem_nada']


list_clfs = ['GaussianNB','KNeighborsClassifier(8)','DecisionTreeClassifier()','RandomForestClassifier','SVM','MLPClassifier']
list_clfs_index = [0,5,6,9,10,11]

main_path = os.getcwd()+'/'

for dataset in list_Datasets:
  for seed_value in list_seeds:
    for mtd_over in list_over:
      for seed_mtd in list_seeds:
        print('Calculando scores para {} seed {} do metodo {} do seed {}'.format(dataset,seed_value,mtd_over,seed_mtd))
        if mtd_over == 'sem_nada':
           path_result = dataset
        else:
            path_result = dataset+'_over_'+mtd_over+'_'+str(seed_mtd)
            
        df = pd.read_csv('seed_'+str(seed_value)+'/'+path_result+'/'+path_result+'_test.csv', index_col=False)
        X = df.drop(['class'], axis=1)
        y = df['class']
        y=list(y.astype('int'))

        df = pd.read_csv('seed_'+str(seed_value)+'/'+path_result+'/'+path_result+'.csv', index_col=False)
        resp_clf = {}
        pred_clf = {}
        for c,clf_index in enumerate(list_clfs_index):
            resp = list(df.loc[clf_index])
            resp_clf[list_clfs[c]] = resp
            pred_clf[list_clfs[c]] = get_classication(resp,y)

        clf_acc = {}
        clf_f1 = {}
        clf_mcc = {}
        for clf in pred_clf:
           clf_acc[clf] = accuracy_score(y,pred_clf[clf])
           clf_f1[clf] = f1_score(y,pred_clf[clf])
           clf_mcc[clf] = matthews_corrcoef(y,pred_clf[clf])
        
        tmp = {}
        for clf in clf_acc:
           tmp[clf] = {'acc_score':clf_acc[clf], 'f1_score':clf_f1[clf], 'MCC':clf_mcc[clf]}
        df = pd.DataFrame.from_dict(tmp).transpose()
        df.to_csv('seed_'+str(seed_value)+'/'+path_result+'/acc_f1_score.csv')

