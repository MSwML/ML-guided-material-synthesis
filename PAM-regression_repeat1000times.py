import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import pearsonr

  
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 
from sklearn.model_selection import KFold, GridSearchCV

import time
import random

from utils import data_handler, plotter

#handle warnings
import warnings
warnings.filterwarnings("ignore")

np.random.seed(3)



def test(y_pred,y_true,end_ptr):
    '''Evaluate the performance of PAM through 3 metrics; 
       And find the best predicted condition for the next PAM trial
       
       Inputs:
           X_test      - for testing
           y_true      - ground true for X_test
           end_ptr     - range of [0,end_ptr] in y_pred and y_true are values observable to PAM
           
       Uses:(global variables)    
           pred_result - of shape (total_samp,), storing all the results predicted by PAM
           true_results- of shape (total_samp,), the same order as of pred_results, storing all the ground truth
           
       Outputs:
            
    
    '''
    #All sample metrics    
    r2 = metrics.r2_score(y_true,y_pred)   
    mse = metrics.mean_squared_error(y_true,y_pred)
    [pear,p_value] = pearsonr(y_true,y_pred)
    
    #metrics based on observable samples
    y_true_s = y_true[0:(end_ptr+1)]
    y_pred_s = y_pred[0:(end_ptr+1)]
    r2_s = metrics.r2_score(y_true_s,y_pred_s)   
    mse_s = metrics.mean_squared_error(y_true_s,y_pred_s)
    [pear_s,p_value_s] = pearsonr(y_true_s,y_pred_s)
    return [r2,pear,p_value,mse,r2_s,pear_s,p_value_s,mse_s] #, best_pos_ind, best_prob 



# data handling
X_df,Y_df = data_handler.load_XY(1)
X = X_df.as_matrix()
Y = Y_df.as_matrix()/100


# setup and construct initial training dataset
# cross validation settup
inner_nsplits = 10
init_train_size = 20
totalSamp = X.shape[0]


Y_global_max = np.max(Y)
all_ind = np.random.permutation(list(range(0,totalSamp)))
all_ind_wo_max = list(range(0,totalSamp))
all_ind_wo_max.remove(0)


# PAM guided synthesis
def PAM_regression(save_csv= False, verbose=False, title='PAM_results_regression',batch=1):

    ## start PAM guided synthesis...
    init_time = time.time()

    #construct initial training set
    results_mat = np.zeros(((totalSamp-init_train_size),12))


    train_ind = random.sample(all_ind_wo_max, init_train_size)
    test_ind = [x for x in all_ind if x not in train_ind]

    # set up result storage to compute eval metrics, in the order of PAM
    #  ignore the initial training set, as it is not determined by PAM
    pred_results = np.zeros(totalSamp-init_train_size)
    true_results = np.zeros(totalSamp-init_train_size)


    # setup the hyperparameter range for tuning
    tuned_parameters = dict(learning_rate=[0.01],
                            n_estimators=[300,500,700], #100,,300,400,500
                            colsample_bylevel = [0.5,0.7,0.9],
                          gamma=[0,0.2], #0,0.1,0.2,0.3,0.4
                          max_depth =[3,7,11], # [3,7,11]]
                          reg_lambda = [0.1,1,10], #[0.1,1,10]
                         # reg_alpha = [1],
                           subsample=[0.4,0.7,1])

    j=0
    loop_count = 0
    mean_y_only_init = np.mean(Y[train_ind])
    std_y_only_init = np.std(Y[train_ind])
    
    
    while(j<totalSamp-init_train_size):        
        inner_cv = KFold(n_splits=inner_nsplits,shuffle=True, random_state=j) 
        X_train = X[train_ind]
        Y_train = Y[train_ind]
        X_test = X[test_ind]
        Y_test = Y[test_ind]

        last_max = np.max(Y_train)
        
        # GradientBoost
        reg = xgb.XGBRegressor(objective="reg:linear",min_child_weight=1,**{'tree_method':'exact'},
                                 silent=True,n_jobs=4,random_state=3,seed=3);

        gb_clf = GridSearchCV(reg,tuned_parameters, cv=inner_cv,scoring='r2',verbose=0,n_jobs=4)
        gb_clf.fit(X_train, Y_train)
        y_pred = gb_clf.predict(X_test)


        # choose the batch of conditions with best predicted yield
        best_pos_ind = np.argsort(-y_pred)[:batch]       
        best_prob = y_pred[best_pos_ind]
        next_ind = np.array(test_ind)[best_pos_ind]        
        
        # update results storage
        train_size = len(Y_train)
        temp = list(range(0,len(y_pred)))
        ind_notbest = [x for x in temp if x not in best_pos_ind]
        
        start_ptr = j
        end_ptr = np.min([start_ptr + batch, totalSamp-init_train_size]) 
        pred_results[start_ptr:end_ptr] = best_prob
        pred_results[end_ptr:totalSamp-init_train_size]= y_pred[ind_notbest]
        true_results[start_ptr:end_ptr] = Y_test[best_pos_ind]
        true_results[end_ptr:totalSamp-init_train_size] = Y_test[ind_notbest]

        pred_metrics = test(pred_results,true_results,end_ptr-1)    

        # calculate results
        next_best_true_ind = next_ind[np.argmax(Y_test[best_pos_ind])]
        next_best_y_true = np.max(Y_test[best_pos_ind])
        result_list = [train_size,next_best_true_ind,next_best_y_true,best_prob[0],] + pred_metrics   
        results_mat[loop_count,:] = np.array(result_list)
                     
        loop_count = loop_count +1
        j = j + batch
                     
        if(verbose):
            print('batch_size = '+str(batch), '  ->',str(next_ind))
            print(gb_clf.best_score_)
            print(gb_clf.best_estimator_)
            
            print(loop_count,'->',j,', best_next_ind=',next_best_true_ind, ' best_Y_true=',"{0:.6f}".format(next_best_y_true),' train_max=',"{0:.6f}".format(last_max),' r2=',pred_metrics[0])

        train_ind = [*train_ind , *next_ind]   
        test_ind = [x for x in test_ind if x not in next_ind]
        
        ## critical point
        if(next_best_y_true==Y_global_max):
            break;


    results = pd.DataFrame(data=results_mat[0:j,:],columns=['sample_size','pred_ind','best_pred_result','y_true','r2','pearson','p_value','mse','r2_s','pearson_s','p_value_s','mse_s'])
    
    if(save_csv):
        data_handler.save_csv(results,title=title)
        
    
    run_time = (time.time() - init_time)/60


    num_experiments = j+init_train_size
    mean_y_wo_init =  np.mean(true_results[0:j])
    std_y_wo_init = np.std(true_results[0:j])
    
    mean_y_w_init = np.mean(Y[train_ind])
    std_y_w_init = np.std(Y[train_ind])
    
    return [num_experiments,mean_y_wo_init,std_y_wo_init,mean_y_w_init,std_y_w_init,mean_y_only_init,std_y_only_init, run_time]


outer_loop = 10
n_loop = 100

# save the results every 100 repetitions for backup
for j in range(0,outer_loop):
    PAM_results = np.zeros((n_loop,8))
    for i in range(0,n_loop):        
        PAM_results[i,:] = np.array(PAM_regression(save_csv= False, verbose=False, title='PAM_results_regression_'+str(i)))
        print(str(i+j*n_loop),' -> ',str(PAM_results[i,0]),'  time=',PAM_results[i,7])
    PAM_df = pd.DataFrame(data=PAM_results, columns=['num_experiments','mean_y_wo_init','std_y_wo_init','mean_y_w_init','std_y_w_init','mean_y_only_init','std_y_only_init', 'run_time'])
    data_handler.save_csv(PAM_df,title='PAM_'+str(n_loop)+'_')
    print('total = ',str((np.sum(PAM_results[:,7])/60)),'  hrs  >>-------saved')
    
print('total = ',str((np.sum(PAM_results[:,7])/60)),'  hrs  >>-------saved')