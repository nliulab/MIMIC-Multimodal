import os
import numpy as np
import pandas as pd
import pickle
import argparse
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score
import torch
from classification_utils import *
import warnings
warnings.filterwarnings("ignore")

# args
parser = argparse.ArgumentParser()

## file path
parser.add_argument('--input-icu-path', type=str, default='Data/processed_icu_24h.pkl')
parser.add_argument('--input-metadata-path', type=str, default='Data/metadata_24h.csv')
parser.add_argument('--embedding-path', type=str,default='Embeddings/')
parser.add_argument('--output-path', type=str,default='Updated Results/')

## Embeddings
parser.add_argument('--time-series-model', type=str,default='gru',help='fixed_time_interval/gru/moment')
parser.add_argument('--image-model', type=str,default='cxr_foundation',help='cxr_foundation/swin_transformer')
parser.add_argument('--text-model', type=str,default='radbert',help='radbert/openai')
parser.add_argument('--outcome', type=str,default='hospital_expire_flag',help='hospital_expire_flag/los_binary/readmission')
parser.add_argument('--classification-model', type=str,default='LR')

args = parser.parse_args()

# Functions
## Concatenate for different embedding techniques and data sets
def cross_concatenation(sets):
    data_dic = {}
    for set in sets:
        # print(set)
        tabular = globals()[f'tabular_embeddings_{set}']
        ts_embedding = globals()[f'ts_embeddings_{set}']
        img_embedding = globals()[f'image_embeddings_{set}']
        txt_embedding = globals()[f'text_embeddings_{set}']
        outcome = globals()[f'outcomes_{set}']
        combinded_df = concatenate(tabular,ts_embedding,img_embedding,txt_embedding,outcome)
        combinded_df.columns = combinded_df.columns.astype(str)
        data_dic[set] = combinded_df
    return data_dic

if __name__ == "__main__":
    
    print(args)
    input_icu_path = args.input_icu_path
    input_metadata_path = args.input_metadata_path
    output_path = args.output_path
    # Load data
    print('Data Preparation')
    icus = pickle.load(open(input_icu_path,'rb'))
    icus_metadata = pd.read_csv(input_metadata_path,index_col=0)
    
    # Train-val-test split
    idx_train_val, idx_test = train_test_split(icus_metadata.index.values, test_size=0.2, random_state=0)
    idx_train,idx_val = train_test_split(idx_train_val, test_size=0.125, random_state=0)
    ## Train
    icus_train = list(itemgetter(*idx_train)(icus))
    metadata_train = icus_metadata.loc[idx_train,:]
    metadata_train = metadata_train.reindex(idx_train)
    ## Validatin
    icus_val = list(itemgetter(*idx_val)(icus))
    metadata_val = icus_metadata.loc[idx_val,:]
    metadata_val = metadata_val.reindex(idx_val)
    ## Test
    icus_test = list(itemgetter(*idx_test)(icus))
    metadata_test = icus_metadata.loc[idx_test,:]
    metadata_test = metadata_test.reindex(idx_test)
    
    # Embeddings
    print('Embeddings')
    embedding_path = args.embedding_path
    outcome = args.outcome
    sets = ['train','val','test']
    ## Tabular
    tabular_embeddings_train = load_npz(os.path.join(embedding_path,'tabular_embeddings_train.npz'))
    tabular_embeddings_val = load_npz(os.path.join(embedding_path,'tabular_embeddings_val.npz'))
    tabular_embeddings_test = load_npz(os.path.join(embedding_path,'tabular_embeddings_test.npz'))
    
    ## Time-series
    time_series_model = args.time_series_model
    if time_series_model == 'fixed_time_interval':
        ts_embeddings = load_npz(os.path.join(embedding_path,f'{time_series_model}_embeddings.npz'))
        ### Train-Val-Test
        ts_embeddings_train = ts_embeddings.loc[idx_train,:]
        ts_embeddings_val = ts_embeddings.loc[idx_val,:]
        ts_embeddings_test = ts_embeddings.loc[idx_test,:]
        ### Standardize
        scaler = StandardScaler()
        cols = ts_embeddings.columns[1:]
        ts_embeddings_train[cols] = scaler.fit_transform(ts_embeddings_train[cols])
        ts_embeddings_val[cols] = scaler.transform(ts_embeddings_val[cols])
        ts_embeddings_test[cols] = scaler.transform(ts_embeddings_test[cols])
        ### Drop stay_id
        ts_embeddings_train.drop(columns=['stay_id'],inplace=True)
        ts_embeddings_val.drop(columns=['stay_id'],inplace=True)
        ts_embeddings_test.drop(columns=['stay_id'],inplace=True)
    elif time_series_model == 'gru':
        # new_embedding_path = 'Updated Embeddings/'
        ts_embeddings_train = load_npz(os.path.join(embedding_path,f'{time_series_model}_{outcome}_embeddings_train.npz'))
        ts_embeddings_val = load_npz(os.path.join(embedding_path,f'{time_series_model}_{outcome}_embeddings_val.npz'))
        ts_embeddings_test = load_npz(os.path.join(embedding_path,f'{time_series_model}_{outcome}_embeddings_test.npz'))
    else:
        ts_embeddings_train = load_npz(os.path.join(embedding_path,f'{time_series_model}_embeddings_train.npz'))
        ts_embeddings_val = load_npz(os.path.join(embedding_path,f'{time_series_model}_embeddings_val.npz'))
        ts_embeddings_test = load_npz(os.path.join(embedding_path,f'{time_series_model}_embeddings_test.npz'))
        
    ## Image
    image_model = args.image_model
    image_embeddings= load_npz(os.path.join(embedding_path,f'{image_model}_embeddings.npz'))
    ### Rename columns
    image_column_names = [f'img_{col}' for col in image_embeddings.columns]
    image_embeddings.columns = image_column_names
    ### Reindex the original embeddings into train,validation and test sets
    image_embeddings_train = image_embeddings.reindex(idx_train)
    image_embeddings_val = image_embeddings.reindex(idx_val)
    image_embeddings_test = image_embeddings.reindex(idx_test)
    
    ## Text
    text_model = args.text_model
    text_embeddings= load_npz(os.path.join(embedding_path,f'{text_model}_embeddings.npz'))
    ### Rename columns
    text_column_names = [f'txt_{col}' for col in text_embeddings.columns]
    text_embeddings.columns = text_column_names
    ### Reindex the original embeddings into train,validation and test sets
    text_embeddings_train = text_embeddings.reindex(idx_train)
    text_embeddings_val = text_embeddings.reindex(idx_val)
    text_embeddings_test = text_embeddings.reindex(idx_test)
    
    ## Outcome
    outcomes_train = metadata_train.loc[:,[outcome]]
    outcomes_val = metadata_val.loc[:,[outcome]]
    outcomes_test = metadata_test.loc[:,[outcome]]
    
    # Concatenate
    data = cross_concatenation(sets)
    print(data['train'].head())
    print(data['val'].head())
    print(data['test'].head())
    # sample_data = {}
    # for set in sets:
    #     sample_data[set] = data[set].iloc[0:100,:]
    
    
    # Model
    print('Model Training')
    
    if args.classification_model == 'LR':
        model = LogisticRegression()
        params = {
            'C': np.logspace(-2, 1, 4),
            'penalty': ['l1', 'l2', None],
            'solver': ['saga'],
            'max_iter': [100,300,500]
            }
    elif args.classification_model == 'LightGBM':
        model = lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0, verbose=-1)
        params = {
            'max_depth': [5, 10, 20], 
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_data_in_leaf': [20, 100, 500],
            'n_estimators': [100, 200]
            }
        
    pred_prob = {}
    fit_models = {}
    
    tabular_column_names = tabular_embeddings_train.columns[3:].to_list()
    time_series_column_names = ts_embeddings_train.columns.astype(str).to_list()
    
    ## Baseline
    print('Baseline')
    ### Load data
    input_variables = tabular_column_names+time_series_column_names
    X_train,y_train,X_test,y_test = load_data(data,input_variables,outcome)
    pred_prob['True label'] = y_test
    ### Train and evaluate
    y_pred_prob,clf_baseline = train_and_predict(model,params,X_train,y_train,X_test)
    pred_prob['Baseline'] = y_pred_prob
    fit_models['Baseline'] = clf_baseline
    ## Tabular+Time-series+Image
    print('Baseline+Image')
    ### Load data
    input_variables = tabular_column_names+time_series_column_names+image_column_names
    X_train,y_train,X_test,y_test = load_data(data,input_variables,outcome)
    ### Train and evaluate
    y_pred_prob,clf_baseline_image = train_and_predict(model,params,X_train,y_train,X_test)
    pred_prob['Baseline+Image'] = y_pred_prob
    fit_models['Baseline+Image'] = clf_baseline_image
    
    ## Tabular+Time-series+Text
    print('Baseline+Text')
    ### Load data
    input_variables = tabular_column_names+time_series_column_names+text_column_names
    X_train,y_train,X_test,y_test = load_data(data,input_variables,outcome)
    ### Train and evaluate
    y_pred_prob,clf_baseline_text = train_and_predict(model,params,X_train,y_train,X_test)
    pred_prob['Baseline+Text'] = y_pred_prob
    fit_models['Baseline+Text'] = clf_baseline_text
    
    ## Tabular+Time-series+Image+Text
    print('Baseline+Image+Text')
    ### Load data
    input_variables = tabular_column_names+time_series_column_names+image_column_names+text_column_names
    X_train,y_train,X_test,y_test = load_data(data,input_variables,outcome)
    print("Training data size:",X_train.shape)
    ### Train and evaluate
    y_pred_prob,clf_baseline_image_text = train_and_predict(model,params,X_train,y_train,X_test)
    pred_prob['Baseline+Image+Text'] = y_pred_prob
    fit_models['Baseline+Image+Text'] = clf_baseline_image_text
    
    # Output performance metrics
    output_path = args.output_path
    result_path = os.path.join(output_path,f'{outcome}/{time_series_model}_{image_model}_{text_model}_results.pkl')
    model_path = os.path.join(output_path,f'{outcome}/{time_series_model}_{image_model}_{text_model}_models.pkl')
    # Export
    pickle.dump(pred_prob,open(result_path,'wb'))
    print(f'Results exported to path {result_path}')
    pickle.dump(fit_models,open(model_path,'wb'))
    print(f'Models exported to path {model_path}')
