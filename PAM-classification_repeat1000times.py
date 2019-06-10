import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from sklearn import metrics 
from sklearn.model_selection import KFold, GridSearchCV,StratifiedKFold


import matplotlib.pyplot as plt

import time

from utils import data_handler, plotter


#handle warnings
import warnings
warnings.filterwarnings("ignore")



def test(clf, X_test,y_test):
    y_pred = clf.predict(X_test)
    c_mat = metrics.confusion_matrix(y_test,y_pred)
    
    best_pos_ind = 0
    fn = 0
    fp = 0
    if(len(c_mat)==1):# all correctly predicted as one class
        print(c_mat)
        best_pos_ind = -1
        if(y_test[0]==0): # all predicted as negative class
            tpr_ts = float('nan')
            tnr_ts = 1
            acc_ts = 1
        if(y_test[0]==1): #all predicted as positive class
            tpr_ts = 1
            tnr_ts = float('nan')   
            acc_ts = 1
    else:
        fn = c_mat[1,0]
        fp = c_mat[0,1]
        tpr_ts = c_mat[1,1]/(c_mat[1,0]+c_mat[1,1])#TP/(TP+FN)
        tnr_ts =  c_mat[0,0]/(c_mat[0,0]+c_mat[0,1])
        acc_ts = (c_mat[0,0] + c_mat[1,1])/c_mat.sum()
    
    pred_prob = clf.predict_proba(X_test)  

    best_prob = float('nan')
    if(best_pos_ind != -1):
        best_pos_ind = np.argmax(pred_prob[:,1])
        best_prob = pred_prob[best_pos_ind,1]
        
    results = [acc_ts,tpr_ts,tnr_ts]
    return results, best_pos_ind, best_prob , fp, fn


# # Data Handling

X_df,Y_df = data_handler.load_XY()
X = X_df.as_matrix()
Y = Y_df.as_matrix()
feature_list = X_df.columns
print(feature_list)

unique, counts = np.unique(Y, return_counts=True) #unique, counts = numpy.unique(a, return_counts=True)
tot_can_count = counts[1]
tot_cnot_count = counts[0]
print(tot_can_count)
print(tot_cnot_count)
print('Can percentage = ',(tot_can_count)/counts.sum())


# # Set up & construct initial dataset
# cross validation settup
inner_nsplits = 10
totalSamp = X.shape[0]


def generate_init_sets():
    '''
         construct initial training/testing dataset
         to make sure each class has at least $inner_nsplits$ samples
    '''

    can_counter = 0
    cnot_counter = 0
    
    # shuffle indexes of data samples
    Y_global_max = np.max(Y)
    all_ind = np.random.permutation(list(range(0,totalSamp)))
    train_ptr = 0

    while(can_counter < inner_nsplits or cnot_counter< inner_nsplits):
        
        next_ind = all_ind[train_ptr]
        train_ptr = train_ptr+1
        
        if(Y[next_ind] ==1):
            can_counter = can_counter +1            
        else:
            cnot_counter = cnot_counter + 1

    ret_dict = {}
    ret_dict['train_ind'] = list(all_ind[0:train_ptr])
    ret_dict['test_ind']  = list(all_ind[train_ptr:len(all_ind)])
    return ret_dict


# # PAM guided sythesis

def PAM_classfication(verbose = True, save_csv = True, to_break=True):
    '''
        PAM of classification problem.
        
        Arguments:
            verbose : Bool. 
            save_csv: Bool. Whether to save detailed results of the PAM into csv file
            to_break: Bool. Whether to reinforce additional stopping condition when critical point is found
        
        Return:
            [Nc, results[Nc,:]] : Nc is the critical point
    '''
    #critical point 
    Nc = 0

    # setup initial sets
    init_sets = generate_init_sets()
    train_ind = init_sets['train_ind']
    test_ind = init_sets['test_ind']
    if(verbose):
        print(train_ind)  
        
    # Results store 
    init_train_size = len(train_ind)
    init_cnot_count = list(Y[train_ind]).count(0) 
    init_can_count = init_train_size - init_cnot_count
    results_mat = np.zeros((totalSamp-init_train_size,8))
    
    # setup hyperparameter range to tune
    tuned_parameters = dict(learning_rate=[0.01],#0.01,0.1,0.2,0.3
                          n_estimators=[100,300,500], #100
                          gamma=[0,0.2,0.4], #0,0.1,0.2,0.3,0.4
                          max_depth =[5,7,9,11], # [4,5,6]
                          reg_lambda = [0.1,1,10], 
                            colsample_bylevel = [0.9],
                            subsample=[0.4,0.7,1])

    
    # start PAM guided synthesis...
    for j in range(totalSamp):#outter_nspliT  
        inner_cv = StratifiedKFold(n_splits=inner_nsplits, shuffle=True,random_state=j) #StratifiedKFold(n_splits=inner_nsplits, random_state=j)
        X_train = X[train_ind]
        Y_train = Y[train_ind]
        X_test = X[test_ind]
        Y_test = Y[test_ind]
        
        #count pos/neg of training set
        tr_zero_count = list(Y_train).count(0)
        tr_total_count = len(train_ind)
        pos_tr = tr_total_count - tr_zero_count

        # GradientBoost
        pipe = xgb.XGBClassifier(objective='binary:logistic',min_child_weight=1,**{'tree_method':'exact'},
                                 silent=True,n_jobs=4,random_state=3,seed=3, scale_pos_weight=1);


        gb_clf = GridSearchCV(pipe,tuned_parameters, cv=inner_cv,scoring='roc_auc',verbose=0,n_jobs=4)
        gb_clf.fit(X_train, Y_train)
        result_list, next_ind, best_prob,fp_ts, fn_ts = test(gb_clf,X_test,Y_test)


        # calculate results
        type1_err = fn_ts / (tot_can_count - init_can_count)
        type2_err = (fp_ts + tr_zero_count - init_cnot_count) / (tot_cnot_count - init_cnot_count)              
        results_mat[j,:] = np.array([tr_total_count] + result_list + [best_prob ,pos_tr, type1_err, type2_err])
        
        next_ind = test_ind[next_ind]
        if(verbose):
            print(j,'loop, next_ind=',next_ind, ' #tr=',tr_total_count,' pos_tr=',pos_tr,' best_prob=',"{0:.6f}".format(best_prob),' type1=',"{0:.6f}".format(type1_err),' type2=',"{0:.6f}".format(type2_err))

        # critical point
        if((best_prob <0.5) and (Nc == 0)):            
            Nc = tr_total_count            
            if(to_break):
                break

        #stopping condition
        if(pos_tr == tot_can_count):
            break
        
        #update train/test sets
        train_ind = train_ind + [next_ind]      
        test_ind.remove(next_ind)

    print('end at loop ',j, '  Nc = ',Nc)
    
    saved_title = 'n'
    if(save_csv):
        results_df = pd.DataFrame(data=results_mat[0:j+1],columns=['sample_size','acc_ts','tpr_ts','tnr_ts','best_prob','pos_tr','type1_err','type2_err'])
        saved_title = data_handler.save_csv(results_df,title='mos2_PAM_results_Nc_'+str(Nc))
        
    return [saved_title, Nc] +results_mat[j].tolist()


# In[8]:

outer_loop = 10
nloop = 100

for j in range(0,outer_loop):
    res_mat = []
    for i in range(0,nloop):
        init_time = time.time()
        arr = PAM_classfication(verbose = False, save_csv = False, to_break=True)
        res_mat.append(arr)
        print(j*nloop + i, ' run time=',str((time.time()-init_time)/60),'mins' )
        print(arr)

    results_df = pd.DataFrame(data= res_mat,columns=['file-name','Nc','sample_size','acc_ts','tpr_ts','tnr_ts','best_prob','pos_tr','type1_err','type2_err'])
    data_handler.save_csv(results_df,title='mos2_PAM_100times_')


