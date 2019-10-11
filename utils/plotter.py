import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
sns.set_style("whitegrid")


import sys
import os
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold,RepeatedKFold, learning_curve
from xgboost.sklearn import XGBClassifier 

from utils import data_handler
from utils import bayesiantests as bt

root_dir = str(Path(os.getcwd())) #.parent
to_dir = root_dir + '/results/'

import warnings
warnings.filterwarnings('ignore')

#res= None
##------------------------------ font, fig size setup------------------------------

plt.rc('font', family='serif')

def set_fig_fonts(SMALL_SIZE=22, MEDIUM_SIZE=24,BIGGER_SIZE = 26):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

set_fig_fonts()

##------------------------------functions----------------------------------------
def save_fig(fig, title):
    to_path = data_handler.format_title(to_dir,title,'.png')
    fig.savefig(to_path ,dpi=1000,bbox_inches="tight",pad_inches=0)#, bbox_inches='tight', pad_inches=10
    print("Successfully saved to: ",to_path)
    return to_path

def plot_correlation_matrix(X,title, col_list, toSaveFig=True):
    
    set_fig_fonts(12,14,16)
    # standardization
    scaler = StandardScaler()
    df_transf = scaler.fit_transform(X)
    df = pd.DataFrame(df_transf,columns = col_list)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('coolwarm', 30)
    #cax = ax1.pcolor(df.corr(), cmap=cmap, vmin=-1, vmax=1)
    mat = df.corr()
    flip_mat = mat.iloc[::-1]

    
    cax = ax1.imshow(flip_mat , interpolation="nearest", cmap=cmap,vmin=-1, vmax=1)
    ax1.grid(True)
    #plt.suptitle('Features\' Correlation', y =0)
    labels=df.columns.tolist()
    x_labels = labels.copy()
    labels.reverse()

    #ax1.xaxis.set_ticks_position('top')
    ax1.set_xticks(np.arange(len(labels)))#np.arange(len(labels))
    ax1.set_yticks(np.arange(len(labels)))  
    # want a more natural, table-like display
    #ax1.xaxis.tick_top()
    ax1.set_xticklabels(x_labels, rotation = -45, ha="left") #, , rotation = 45,horizontalalignment="left"
    ax1.set_yticklabels(labels, ha="right")
    
    #plt.xticks(rotation=90)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, boundaries=np.linspace(-1,1,21),ticks=np.linspace(-1,1,5))
    plt.show()
    if(toSaveFig):
        save_fig(fig,title+'_confusion_matrix')
    set_fig_fonts()

    
def plot_ROC_curve(pipe, tuned_parameters, title = 'roc_curve', save_csv = True,task=0):
    # cross validation settup
    Ntrials = 1
    outter_nsplit = 10
    inner_nsplit = 10

    # Results store
    Y_true = pd.Series(name='Y_true')
    pred_results = pd.Series(name='pred_prob')

    # load data
    assert (task ==0 or task ==2),'Error: invalid task spec!'
    X_df, Y_df = data_handler.load_XY(task)
    X = X_df.values
    Y = Y_df.values

    for i in range(Ntrials):
        
        train_index = []  
        test_index = []  


        outer_cv = StratifiedKFold(n_splits=outter_nsplit, shuffle=True, random_state=i)
        for train_ind,test_ind in outer_cv.split(X,Y):
            train_index.append(train_ind.tolist())
            test_index.append(test_ind.tolist())

        for j in range(outter_nsplit):#outter_nsplit
            print("progress >> ",j,' / ',outter_nsplit)
            X_train = X[train_index[j]]
            Y_train = Y[train_index[j]]

            X_test = X[test_index[j]]
            Y_test = Y[test_index[j]]


            inner_cv = StratifiedKFold(n_splits=inner_nsplit, shuffle=False, random_state=j)

            clf = GridSearchCV(pipe,tuned_parameters, cv=inner_cv,scoring='roc_auc')
            clf.fit(X_train, Y_train)
            pred = pd.Series(clf.predict_proba(X_test)[:,1])
            pred_results = pd.concat([pred_results, pred], axis=0,ignore_index=True)
            Y_test_df = pd.Series(Y_test,name='Y_test')
            Y_true = pd.concat([Y_true,Y_test_df], axis=0,ignore_index=True)
    
    
    # plotting
    fpr, tpr, thresholds = metrics.roc_curve(Y_true,pred_results)
    roc_auc = metrics.auc(fpr, tpr)
    auc_value = metrics.roc_auc_score(Y_true, pred_results)

    fig = plt.figure(figsize=(12,12/1.618))
    ax1 = fig.add_subplot(111)

    labl = np.linspace(0,1,6)
    labels = [float("{0:.2f}".format(x)) for x in labl]

    ax1.set_xticks(labels)
    ax1.set_xticklabels(labels) 
    labels[0] = ''
    ax1.set_yticklabels(labels)
    plt.grid(False)

    ax1.plot(fpr, tpr, lw=2, label='ROC curve (area = {:.2f})'.format(auc_value),marker='.', linestyle='-', color='b')
    ax1.plot([0,1],[0,1], linestyle='--', color='k')

    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0,1)
    ax1.legend(loc='lower right')

    color = 'black'
     
    plt.setp(ax1.spines.values(), color=color)
    ax1.yaxis.set_visible(True)
    ax1.xaxis.set_visible(True)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.get_yaxis().set_tick_params(direction='out', width=2)
    plt.show()
    fig.savefig(data_handler.format_title(to_dir,title+'_ROC_curve','.png'),dpi=1000,bbox_inches="tight",pad_inches=0)
    
    # save results to csv if true
    if save_csv:
        data_mat = np.array([fpr,tpr]).T
        ret = pd.DataFrame(data_mat,columns=['fpr','tpr'])
        data_handler.save_csv(ret,title+'_ROC_curve')
    return True;


 
    
    
def plot_learning_curve_versus_tr_epoch(title='',ntrials=1, nfolds=10, save_csv=False,verbose=True, save_fig=False): 
    X_df,Y_df = data_handler.load_XY()
    X = X_df.values
    Y = Y_df.values


    _ylabel = 'Mean AUROC'
    n_jobs=4

    # cross validation settup
    Ntrials = ntrials
    outter_nsplit = nfolds
    tot_count = Ntrials * outter_nsplit

    # Results store
    train_mat = np.zeros((tot_count,500))
    test_mat = np.zeros((tot_count,500))



    for i in range(Ntrials):
        init_time = time.time()
        print("trial = ",i)
        train_index = []  
        test_index = []  

        outer_cv = StratifiedKFold(n_splits=outter_nsplit, shuffle=True, random_state=i)
        for train_ind,test_ind in outer_cv.split(X,Y):
            train_index.append(train_ind.tolist())
            test_index.append(test_ind.tolist())


        for j in range(outter_nsplit):#outter_nsplit
            count = i * outter_nsplit + j
            print(str(count), "  / ",str(tot_count))
            X_train = X[train_index[j]]
            Y_train = Y[train_index[j]]

            X_test = X[test_index[j]]
            Y_test = Y[test_index[j]]

            eval_sets = [(X_train, Y_train), (X_test,Y_test)]

            clf = XGBClassifier(objective="binary:logistic",min_child_weight=1,**{'tree_method':'exact'},silent=True,
                                n_jobs=4,random_state=3,seed=3,
                                    learning_rate=0.01,
                                    colsample_bylevel=0.9,
                                    colsample_bytree=0.9,
                              n_estimators=500,
                              gamma=0.8,
                              max_depth =11, 
                              reg_lambda = 0.8,
                                   subsample=0.4)
            clf.fit(X_train,Y_train, eval_metric=['auc'], eval_set = eval_sets, verbose=False)
            results = clf.evals_result()
            epochs = len(results['validation_0']['auc'])

            # record results
            train_mat[count] = results['validation_0']['auc']
            test_mat[count] = results['validation_1']['auc']


            if(verbose):
                print('Iter: %d, epochs: %d'%(count, epochs))
                print('training result: %.4f, testing result: %.4f'%(train_mat[count][499], test_mat[count][499]))


   
        print('total time: %.4f mins'% ((time.time()-init_time)/60))

        
    # Results store
    epoch_lists=list(range(1,epochs+1))
    train_results = pd.DataFrame(data=train_mat,columns=['epoch_'+str(i) for i in epoch_lists])
    test_results = pd.DataFrame(data=test_mat,columns=['epoch_'+str(i) for i in epoch_lists])

    if(save_csv):
        data_handler.save_csv(train_results,title='mos2_learning_curve_train_raw')
        data_handler.save_csv(test_results,title='mos2_learning_curve_test_raw')


    print('end')

    _ylim=(0.5, 1.01)
    n_jobs=4
    

    # create learning curve values
    train_scores_mean = np.mean(train_mat, axis=0)
    train_scores_std = np.std(train_mat, axis=0)
    test_scores_mean = np.mean(test_mat, axis=0)
    test_scores_std = np.std(test_mat, axis=0)

    tr_size_df = pd.Series(epoch_lists, name='training_epoch')
    tr_sc_m_df = pd.Series(train_scores_mean, name='training_score_mean')
    val_sc_m_df = pd.Series(test_scores_mean, name='val_score_mean')
    tr_sc_std_df = pd.Series(train_scores_std, name='training_score_std')
    val_sc_std_df = pd.Series(test_scores_std, name='val_score_std')

    if(save_csv):
        res = pd.concat([tr_size_df, tr_sc_m_df,val_sc_m_df,tr_sc_std_df,val_sc_std_df], axis=1)
        data_handler.save_csv(data=res,title=title+'_learning_curve')

    
    # plotting
    _ylim=(0.5, 1.01)

    fig = plt.figure(figsize=(12,12/1.618))
    ax1 = fig.add_subplot(111)

    ax1.set_ylim(_ylim)
    ax1.set_xlabel("Number of Training Epochs")
    ax1.set_ylabel(_ylabel)
    plt.grid(False)

    ax1.plot(tr_size_df, tr_sc_m_df,  color="r", label="Training") #'o-',
    ax1.plot(tr_size_df, val_sc_m_df,  color="b", label="Validation") #'^--',
    # plot error bars
    #ax1.errorbar(tr_size_df, tr_sc_m_df, yerr=tr_sc_std_df,color="r", )
    #ax1.errorbar(tr_size_df, val_sc_m_df, yerr=val_sc_std_df)

    plt.setp(ax1.spines.values(), color='black')
    plt.legend(loc="lower right")

    plt.show()
    to_path = None
    if save_fig:
        to_path = data_handler.format_title(to_dir,title+'_learning_curve','.png')
        fig.savefig(to_path,dpi=1000,bbox_inches="tight",pad_inches=0.1)
    
    return to_path    
    
def plot_learning_curve_versus_tr_set_size(title='',save_csv=True,scoring = 'roc_auc'):
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set
    X, Y = data_handler.load_XY()


    _ylabel = 'Mean AUROC'
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=3)
    xgb_clf = XGBClassifier(objective="binary:logistic",min_child_weight=1,**{'tree_method':'exact'},
                             silent=True,n_jobs=1,random_state=3,seed=3);
    tuned_parameters = dict(learning_rate=[0.01,0.1],
              n_estimators=[100, 300, 500],
              colsample_bylevel = [0.5,0.7,0.9],
              gamma=[0,0.2,0.4],
              max_depth =[3,5,7],
              reg_lambda = [0.1,1,10],
              subsample=[0.4,0.7,1])
    xgb_cv = GridSearchCV(xgb_clf,tuned_parameters, cv=inner_cv,scoring='roc_auc',verbose=0,n_jobs=1)

    _ylim=(0.5, 1.01)
    n_jobs=4
    train_sizes=np.linspace(.2, 1.0, 5)

    # create learning curve values
    train_sizes, train_scores, test_scores = learning_curve(
            xgb_cv, X, Y, cv=outer_cv, n_jobs=4, train_sizes=train_sizes,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    tr_size_df = pd.Series(train_sizes, name='training_set_size')
    tr_sc_m_df = pd.Series(train_scores_mean, name='training_score_mean')
    cv_sc_m_df = pd.Series(test_scores_mean, name='cv_score_mean')
    tr_sc_std_df = pd.Series(train_scores_std, name='training_score_std')
    cv_sc_std_df = pd.Series(test_scores_std, name='cv_score_std')

    if(save_csv):
        res = pd.concat([tr_size_df, tr_sc_m_df,cv_sc_m_df,tr_sc_std_df,cv_sc_std_df], axis=1)
        data_handler.save_csv(data=res,title=title+'_learning_curve')

    
    # plotting
    _ylim=(0.5, 1.01)

    fig = plt.figure(figsize=(12,12/1.618))
    ax1 = fig.add_subplot(111)

    ax1.set_ylim(_ylim)
    ax1.set_xlabel("Number of Training Samples")
    ax1.set_ylabel(_ylabel)
    plt.grid(False)

    ax1.plot(tr_size_df, tr_sc_m_df, 'o-', color="r", label="Training")
    ax1.plot(tr_size_df, cv_sc_m_df, '^--', color="b", label="Cross-Validation")

    plt.setp(ax1.spines.values(), color='black')
    plt.legend(loc="lower right")

    plt.show()
    to_path = data_handler.format_title(to_dir,title+'_learning_curve','.png')
    fig.savefig(to_path,dpi=1000,bbox_inches="tight",pad_inches=0)
    
    return to_path
    
    
    
def plot_boxplots(data=None, ylabels = [], xmin =0.2, xmax= 1.025,toSaveFig = True, title='boxplot',
                       palette_colors=sns.xkcd_palette(["orange","yellow", "medium green","windows blue"])):
    '''
    Plot boxplots. Stack subplots horizontally.
    
    Arguments:
        data: List of dataframes. Length of data is the number of subplots, while each data array is inputs to each subplot.
        ylabels: List of strings. Y-labels of each subplots. Must be same length of $data$.
        xmin: Float or List. Min of x-axises.
        xmax: Float or List. Max of x-axises.
        toSaveFig: Bool. Whether to save the figure.
        title: String. If $toSaveFig$ is True, save the figure with filename of $title$
        palette_colors: palette colors of seaborn.

    
    '''
    #palette_colors2 = sns.color_palette(["#FF8000", "#FFFF00", "#00FF00", "#3333FF"])
    set_fig_fonts(20,24,26)
    # Create a figure instance, and the two subplots
    #xmin = 0.2
    #xmax = 1.025
    assert (len(data) ==len(ylabels)), "Error: len(data)!=len(names)"   
    n_subplot = len(ylabels)
    
    # handle xmin, xmax
    import numbers
    if isinstance(xmin, numbers.Number):
        xmin = [xmin] * len(ylabels)
    if isinstance(xmax, numbers.Number):
        xmax = [xmax] * len(ylabels)
    assert (len(xmin) ==len(ylabels)), "Error: len(xmin)!=len(names)" 
    assert (len(xmax) ==len(ylabels)), "Error: len(xmax)!=len(names)"  
    
    # start drawing
    fig = plt.figure(figsize=(12,15))
    
    for i in range(n_subplot):
        
        # add each subplot to fig        
        ax = fig.add_subplot(n_subplot,1,i+1)
        ax = sns.boxplot(data=data[i], orient="h", palette=palette_colors)#plot_kws={'line_kws':{'alpha':0.5}
        ax.xaxis.tick_top()
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(xmin[i],xmax[i])
        
        # if not the first subplot, switch off x-labelling
        if(i!=0):
            ax.tick_params(labelbottom='off')  
            
        plt.setp(ax.spines.values(), color='black')

    plt.show()
    
    if(toSaveFig):
        title = title+'_boxplot'
        save_fig(fig,title)
    set_fig_fonts()
        

    
def plot_ttest(x, rope, verbose=False, names=('C1', 'C2'),runs=1,nsamples=50000,title=None,toSaveFig = False, x_lims = (-0.07,0.37) ):
    '''
    Plot posterior distribution for Bayesian correlated t-test with MCMC sampling.
    
    Arguments:
        x: Array of float. data
        rope: Float.
        runs: Int. #runs of k-fold cross validation.
        nsamples: Int.
        title: String. If $toSaveFig$ is True, save the figure with filename of $title$
        toSaveFig: Bool. Whether to save the figure.
    '''
    #--- 1) call 'correlated_ttest'
    left, within, right = bt.correlated_ttest(x=x, rope=rope, runs=runs, verbose=verbose, names=names)

    #--- 2) plot posterior    
    #generate samples from posterior (it is not necesssary because the posterior is a Student)
    samples=bt.correlated_ttest_MC(x, rope=rope,runs=runs,nsamples=nsamples)
    #plot posterior
    fig = sns.kdeplot(samples, shade=True) 
    #plot rope region
    plt.axvline(x=-rope,color='orange')
    plt.axvline(x=rope,color='orange')
    # align x axis
    axes = fig.axes
    axes.set_xlim(x_lims)
    axes.set_ylim(0,25)
    #add label
    plt.xlabel(title)
    
    if(toSaveFig):
        save_fig(fig.get_figure(),title+'_ttest')
        
    return left, within, right