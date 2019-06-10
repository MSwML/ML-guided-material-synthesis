import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import datetime
import time

root_dir = str(Path(os.getcwd()))
from_dir = root_dir + '/data/'
to_dir = root_dir + '/results/'

##-----complementary functions----------
def dummy_contrast_encode(df1,feature_list1):
    # dummy contrast encoding
    # - convert the categorical features into numerical
    # - http://appliedpredictivemodeling.com/blog/2013/10/23/the-basics-of-encoding-categorical-data-for-predictive-models
    # - https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html#conversion-from-categorical-to-numeric-variables
    new_df = pd.DataFrame()
    feature_dummies = {}

    for col in feature_list1:
        if(str(df1[col].dtypes) == 'object'):
            #handling
            temp = pd.get_dummies(df1[col], columns=[col],prefix = col)
            del temp[temp.columns[0]]
            feature_dummies[col] = list(temp.columns.values)

            for new_col in temp.columns:
                new_df[new_col] = temp[new_col]
                #X_df = X_df.assign(str(new_col)=temp[new_col].values)
        else:
            temp = df1[col]
            new_df[col] = temp

    if (str(df1[df1.columns[len(df1.columns)-1]].dtypes)=='object'):
        le = LabelEncoder()
        le.fit(['Cannot','Can'])
        le.transform(['Can','Cannot', 'Cannot']) 
        temp = 1- le.transform(df1[df1.columns[len(df1.columns)-1]])
        new_df['Result']=temp
    else:
        new_df['Result']= df1['Result']
    return new_df



def load_data(task=0):
    '''
    Load csv file into dataframes.
    '''
    
    assert (task==0 or task==1 or task ==2), 'Error: invalid task spec'
    

    if(task==0):
        print('loading MoS2 dataset...')
        df1 = pd.read_csv(from_dir+'mos2_raw.csv')
        feature_list1 = df1.columns[0:(df1.shape[1]-1)]
        df = dummy_contrast_encode(df1,feature_list1)
    elif(task ==1):
        print('loading CQD dataset...')
        df = pd.read_csv(from_dir+'cqd_raw.csv')
    
    return df;
    
    
def load_XY(task=0, filename=None):
    '''Load dataset into X,Y.
    
    Params:
        task: Integer. Specify which dataset to load
        filename: String. If task is None, load data from the path.
    '''
    if task is not None:
        assert (task==0 or task==1 or task ==2), 'Error: invalid task spec' 
        df = load_data(task)
    else:
        print('loading new CQD dataset...')
        df = pd.read_csv(from_dir+filename)
       
    feature_list = df.columns[0:len(df.columns)-1]
    result_col = df.columns[len(df.columns)-1]

    X = df[feature_list]
    Y = df[result_col]
    
    return X,Y
    


def load_csv(title,isData=True):
    if(isData):
        result = pd.read_csv(from_dir+title+'.csv')
    else:
        result = pd.read_csv(to_dir+title+'.csv')
    return result;
        

def save_csv(data,title,ind=False):
    to_save_title = format_title(to_dir, title, fileEtd='.csv')
    data.to_csv(to_save_title,index=ind)
    print('Successfully saved :',to_save_title)
    return to_save_title;

def update_title_w_date(title):
    now_time = datetime.datetime.now()
    today = str(now_time.year)+'_'+str(now_time.month)+'_'+str(now_time.day)
    return title + today

def format_title(to_dir, title, fileEtd):
    title = update_title_w_date(title)
    to_save_title = to_dir+title+fileEtd
    
    i=0
    while(os.path.exists(to_save_title)):
        to_save_title = to_dir+title+'_'+str(i)+fileEtd
        i= i+1
    return to_save_title

    
def load_fake_input(task=0):
    if(task==0):
        return load_csv('fake_input_mos2')
    elif(task==1):
        return load_csv('fake_input_cqd')

    
def fake_input_generator(feature_list,task=0,  to_save = True):
    ''' generate fake inputs for optimization tasks
        
        Arguments:
            task: {0,1,2}. 0 for mos2, 1 for hydrothermal
            feature_list: name list of conditions to generate and store
            to_save: Bool. If true, save to csv
    '''
    count = 0 # count the total number of generator conditions
    init_time = time.time()

    
    if(task==0):
        addNaCl = [0,1]
        sdist = list_generator(0.5,3.5,0.5)
        #boatQuartz = [0,1]
        faceD_tiled = [0,1]
        flow1 = list_generator(10,250,10)
        temp1 = list_generator(500,1000,25)
        rampT = list_generator(10,20,1)
        delay1 = list_generator(5,15,5)
        #pressure = [100]   
        
        tot = len(addNaCl)*len(sdist)*len(faceD_tiled)*len(flow1)*len(temp1)*len(rampT)*len(delay1);
        mat = np.zeros((tot,len(feature_list)))
        
        print('total =',tot)
        for a in addNaCl: #
            print("{0:.2f}".format(count/tot * 100),'%, time = ',(time.time() - init_time)/60,' mins')
            for s in sdist:
                #for b in boatQuartz: #
                for ft in faceD_tiled: #
                    for f in flow1:
                        for t in temp1:
                            for r in rampT:
                                 for d in delay1:
                                        #for p in pressure:
                                        #new_rec = pd.DataFrame([[a,s,ft,f,t,r,d]],columns=feature_list)
                                        #generated = pd.concat([generated,new_rec],ignore_index=True)
                                        mat[count,:] = np.array([a,s,ft,f,t,r,d])
                                        count = count + 1       
      
    elif(task==1):

        massA = list_generator(0.2,5,0.2)
        massA.extend(list_generator(6,12,1))
        solVol = list_generator(10,60,5)
        reatTemp = list_generator(140,260,10)
        reatT = list_generator(1,9,0.5)
        #reatT.extend(list_generator(0.1,0.9,0.1))
        #reatT.extend(list_generator(1,21,0.5))
        rampR = [2,5,10,15]
        ph = list_generator(5,9,1)

        
        tot = len(massA)*len(solVol)*len(reatTemp)*len(reatT)*len(rampR)*len(ph);
        mat = np.zeros((tot,len(feature_list)))
        print('total = ',tot)
        for m in massA: #
            print("{0:.2f}".format(count/tot *100),'% ,  time = ',(time.time() - init_time)/60,' mins')
            print('m=',m)
            for s in solVol:
                for r in reatTemp: #
                    for rt in reatT: #
                        for rr in rampR:
                            for p in ph:
                                mat[count,:] = np.array([m,s,r,rt,rr,p])
                                count = count + 1
    elif(task==2):
        source_rt = list_generator(1,20,1)
        tmp = list_generator(300,850,50)
        rmp_rate = list_generator(7,27,1)
        flow = list_generator(0,50,5)
        dp_time = list_generator(4,40,3)
        
        tot = len(source_rt)*len(tmp)*len(rmp_rate)*len(flow)*len(dp_time)
        mat = np.zeros((tot,len(feature_list)))
        print('total = ',tot)
        
        for sr in source_rt: #
            print("{0:.2f}".format(count/tot *100),'% ,  time = ',(time.time() - init_time)/60,' mins')
            print('sr=',sr)
            for t in tmp:
                for rr in rmp_rate:
                    for f in flow: #
                        for d in dp_time:
                            mat[count,:] = np.array([sr,t,rr,f,d])
                            count = count + 1
    elif(task==3):

        temp = list_generator(100,220,10)
        tm = list_generator(2,12,1)
        rampR = list_generator(5,30,1)
        prec = list_generator(0.02,0.1,0.01)
        yea = list_generator(0,200,10)

        tot = len(temp)*len(tm)*len(rampR)*len(prec)*len(yea)
        mat = np.zeros((tot,len(feature_list)))
        print('total = ',tot)
        for tp in temp: #
            print("{0:.2f}".format(count/tot *100),'% ,  time = ',(time.time() - init_time)/60,' mins')
            print('temp=',tp)
            for t in tm:
                for r in rampR: #
                    for p in prec: #
                        for y in yea:
                            mat[count,:] = np.array([tp,t,r,p,y])
                            count = count + 1
        
    generated = pd.DataFrame(data=mat,columns = feature_list)                              
    if(to_save):
        if(task==0):
            save_csv(generated, title='fake_input_mos2')
            #save_hugearr_tocsv(arr,feature_list,title='fake_input_mos2'+today)
        elif(task==1):
            save_csv(generated, title='fake_input_cqd')
            #save_hugearr_tocsv(arr,feature_list,title='fake_input_hydro'+today)
        elif(task==2):
            save_csv(generated, title='fake_input_WTe2')

    print(count)
    return generated;

def save_hugearr_tocsv(arr,feature_list,chunksize=1000000,title='no_title'):#1mil
    arr_size = arr.shape[0]
    numchk = int(arr_size /chunksize)
    
    
    for i in range(0,numchk):
        start_ind = chunksize * i;
        end_ind = chunksize *(i+1);
        df = pd.DataFrame(data = arr[start_ind:end_ind,:], columns=feature_list) # Fake inputs store
        save_csv(df, title=title+str(i))
        
    return i;
    
def list_generator(start, end, step):
    list_ = list(range(int(start*100),int(end*100),int(step*100))) + [end*100]
    results = [float("{0:.2f}".format(x*0.01)) for x in list_]
    return(results)