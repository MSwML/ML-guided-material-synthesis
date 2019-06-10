import shap
import numpy as np
import pandas as pd
from utils import data_handler



def extract_feature_importance(model,X,title):
    
    print('Feature importance...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, feature_names=X.columns, plot_type="bar")

    # normalize importance values
    sum_col = abs(shap_values).sum(axis=0)
    imp = np.array(sum_col/sum_col.sum())

    ind = np.argsort(imp)[::-1]
    sorted_imp = imp[ind]
    sorted_feature = X.columns[ind]

    feature_imp_sorted = pd.DataFrame( [sorted_imp],columns=sorted_feature)
    
    
    print(feature_imp_sorted)
    data_handler.save_csv(feature_imp_sorted,title=title+'feature_imp_sorted')
    
    
    
def predict_fake_input(model, task, title):
    
    generated = data_handler.load_fake_input(task)
    print('Number of generated conditions : ',generated.shape)
    
    if(task==0):
        pred = model.predict_proba(generated)
        final_state = pd.Series( pred[:,1], name='Pred_Result')
    elif(task==1):
        pred = model.predict(generated)
        final_state = pd.Series( pred, name='Pred_Result')

    result = pd.concat([generated, final_state], axis=1)
    data_handler.save_csv(result,title+'pred_fake_input')