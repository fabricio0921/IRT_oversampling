import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SVMSMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids
import decodIRT_OtML as dIRT_OtML
import decodIRT_MLtIRT as dIRT_MLtIRT

#LISTAR O(S) DATASETS

list_Datasets = ['pc4']#,'climate-model-simulation-crashes','credit-approval']
np.object=object
list_seeds = [10]
list_seeds_mtd = [10]

#LISTAR A(S) TÉCNICAS A SEREM ANALIZADAS

#list_over = ['adasyn','smote']
list_over = ['adasyn','smote','smoten','svmsmote']

list_under = []

main_path = os.getcwd()+'/'

for dataset in list_Datasets:
  for seed_value in list_seeds:

    out_path = 'seed_'+str(seed_value)
    dIRT_OtML.main(arg_data=dataset,arg_dataset=None,arg_dataTest=None,
               arg_saveData=True,arg_seed=seed_value,arg_output=out_path)
    
    data_test_path = out_path+'/'+dataset+'/'+dataset+'_test.csv'

    df = pd.read_csv(out_path+'/'+dataset+'/'+dataset+'_train.csv', index_col=False)
    X = df.drop(['class'], axis=1)
    y = df['class']
    y=y.astype('int')

    for seed_mtd in list_seeds_mtd:

      dict_over = {'adasyn':ADASYN(random_state=seed_mtd),'smote':SMOTE(random_state=seed_mtd),'smoten':SMOTEN(random_state=seed_mtd),'svmsmote':SMOTEN(random_state=seed_mtd)}
      dict_under = {'nearmiss':NearMiss(),'ClusterCentroids':ClusterCentroids(random_state=seed_mtd)}

      for over_mtd in list_over:
        #ros = ADASYN(random_state=10) # String
        X_over, y_over = dict_over[over_mtd].fit_resample(X, y)

        X_over['class'] = list(y_over)
        save_csv = main_path+dataset+'_over_'+over_mtd+'_'+str(seed_mtd)+'.csv'
        X_over.to_csv(save_csv,index=False)

        dIRT_OtML.main(arg_data=None,arg_dataset=save_csv,arg_dataTest=data_test_path,
                arg_saveData=True,arg_seed=seed_value,arg_output=out_path)

      for under_mtd in list_under:
        #ros = ADASYN(random_state=10) # String
        X_under, y_under = dict_under[under_mtd].fit_resample(X, y)

        X_under['class'] = list(y_under)
        save_csv = main_path+dataset+'_under_'+under_mtd+'_'+str(seed_mtd)+'.csv'
        X_under.to_csv(save_csv,index=False)

        dIRT_OtML.main(arg_data=None,arg_dataset=save_csv,arg_dataTest=data_test_path,
                arg_saveData=True,arg_seed=seed_value,arg_output=out_path)
    
    dIRT_MLtIRT.main(arg_dir = out_path)
  
  
  ############################
    
    dIRT_MLtIRT.main(arg_dir = './seed_10')
    
    ############################
    list_param = ['Discriminacao','Dificuldade','Adivinhacao']

dict_seed = {}
for seed_value in list_seeds:
  dir_path = './seed_'+str(seed_value)
  list_path = os.listdir(dir_path)
  list_path = [i for i in list_path if '.' not in i]
  dict_tecnicas = {}
  for path in list_path:
    data_csv = dir_path+'/'+path+'/irt_item_param.csv'

    tabela = pd.read_csv(data_csv)
    dict_param = {}
    for param in list_param:
      dict_param[param] = {}
      valores = list(tabela[param])
      #MEDIA
      media = np.mean(valores)
      #MEDIANA
      mediana = np.median(valores)
      #DESVIO PRADRÃO
      desvio = np.std(valores)

      dict_param[param]['media'] = media
      dict_param[param]['mediana'] = mediana
      dict_param[param]['desvio'] = desvio

    dict_tecnicas[path] = dict_param
  dict_seed['seed_'+str(seed_value)] = dict_tecnicas
  
  
  ########################################################
  
  import matplotlib.pyplot as plt

for seed_v in dict_seed:
  dataset = [i for i in dict_seed[seed_v]]
  dataset.sort()
  for dt in dataset:
    for param in list_param:
      f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8, 6))
      # For Sine Function
      #print(dict_seed[seed_v][dt][param]['media'])
      data = dict_seed[seed_v][dt][param]['media']
      data = float("{:.3f}".format(data))
      ax1.bar(['media'], data, color='red',alpha=0.6)
      #ax1.set_title(dt+" - "+param)
      ax1.text(x=['media'] , y =data , s=f"{data}" , fontdict=dict(fontsize=12))

      # For Cosine Function
      data = dict_seed[seed_v][dt][param]['mediana']
      data = float("{:.3f}".format(data))
      ax2.bar(['mediana'], data, color='blue',alpha=0.6)
      ax2.text(x=['mediana'] , y =data , s=f"{data}" , fontdict=dict(fontsize=12))
      ax2.set_title(dt+" - "+param)

      # For Tangent Function
      data = dict_seed[seed_v][dt][param]['desvio']
      data = float("{:.3f}".format(data))
      ax3.bar(['desvio'], data, color='green',alpha=0.6)
      ax3.text(x=['desvio'] , y =data , s=f"{data}" , fontdict=dict(fontsize=12))
      #ax3.set_title(dt+" - "+param)
      #plt.show()