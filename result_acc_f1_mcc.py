
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SVMSMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef,classification_report
import decodIRT_OtML as dIRT_OtML
import decodIRT_MLtIRT as dIRT_MLtIRT

import sys
from IPython.display import display
from openpyxl import Workbook


sd = 10
dataset = 'pc4'
tecnica = 'smote'
#LENDO AS TABELAS COM AS INFORMAÇÕES DE ACURÁCIA E F1 SCORE
#data_10= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_10/acc_f1_score.csv")
#data_20= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_20/acc_f1_score.csv")
#data_30= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_30/acc_f1_score.csv")


#LENDO AS TABELAS COM AS INFORMAÇÕES DE ACURÁCIA E F1 SCORE

############################
print('@@@@@@@@@@ COM BALANCEAMNETO @@@@@@@@@@@')
print('')
data_10= pd.read_csv("/home/fabricio/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_10/acc_f1_score.csv")
#data_20= pd.read_csv("/home/fabricio/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_20/acc_f1_score.csv")
#data_30= pd.read_csv("/home/fabricio/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_30/acc_f1_score.csv")
#print('')

#####sem###################
#print('@@@@@@@@@@ SEM BALANCEAMNETO @@@@@@@@@@@')
#print('')
#data_10= pd.read_csv("/home/fabricio/dados_irt/seed_"+str(sd)+"/"+dataset+"/acc_f1_score.csv")
#data_20= pd.read_csv("/home/fabricio/dados_irt/seed_"+str(sd)+"/"+dataset+"/acc_f1_score.csv")
#data_30= pd.read_csv("/home/fabricio/dados_irt/seed_"+str(sd)+"/"+dataset+"/acc_f1_score.csv")





#TRANSORMANDO AS TABELAS EM DATAFRAME
df10 = pd.DataFrame(data_10)
#df20 = pd.DataFrame(data_20)
#df30 = pd.DataFrame(data_30)

#MOSTRANDO O DATAFRAME

seedscc = [df10] #df20, df30...


#MOSTRANDO A COLUNA DE ACURACIA

#MOSTRANDO A MÉDIA DAS ACURÁCIAS
for seeds in seedscc:
            
            #print('##########################')
            #print([seeds])
            #print('######## acuracia ############')
            #print(seeds['acc_score'])
            #print('')
            print('------ MÉDIA acuracia --------')
            print('A média acc_score é: ',seeds["acc_score"].mean())
            print('##############################################')
            print('')
            print('------ MÉDIA F1_SCORE ---------')           

            print('A média F1_score é: ',seeds["f1_score"].mean())
            print('##############################################')
            print('')
            
            print('')
            print('-------- MÉDIA MCC ----------')           

            print('A média MCC é: ',seeds["MCC"].mean())
            print('##############################################')
            print('')

            


           
            

            

    
        
    
#print('A média geral é acc_score é: ', medias["acc_score"].mean())



