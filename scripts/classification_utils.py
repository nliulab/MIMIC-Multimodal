import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

# Embedding concatenation          
def concatenate(tabular,time_series_embedding,image_embedding,text_embedding,outcome):
    data = pd.merge(tabular, time_series_embedding, how='left', left_index=True, right_index=True)
    data = pd.merge(data, image_embedding, how='left', left_index=True, right_index=True)
    data = pd.merge(data, text_embedding, how='left', left_index=True, right_index=True)
    data = pd.merge(data, outcome, how='left', left_index=True, right_index=True)
    return data

# Train-Test Data
def load_data(data_dic,input_variables,outcome_variable):
    
    # Train & Validation
    df_train = data_dic['train'].loc[:,input_variables]
    label_train = data_dic['train'][outcome_variable]
    df_val = data_dic['val'].loc[:,input_variables]
    label_val = data_dic['val'][outcome_variable]
    # Combine train and validation set
    X_train = pd.concat([df_train,df_val])
    X_train.reset_index(drop=True,inplace=True)
    y_train = pd.concat([label_train,label_val])
    y_train.reset_index(drop=True,inplace=True)
    # Test
    X_test = data_dic['test'].loc[:,input_variables]
    X_test.reset_index(drop=True,inplace=True)
    y_test = data_dic['test'][outcome_variable]
    y_test.reset_index(drop=True,inplace=True)
    # Missing data imputation
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train,y_train,X_test,y_test

"""
Model
"""
def train_and_predict(model,parameters,X_train,y_train,X_test):
    clf = GridSearchCV(model, parameters, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    # Fit model
    clf.fit(X_train, y_train)
    # Best model after grid search
    # print("Best parameters found: ", clf.best_params_)
    # print("Best cross-validated score: {:.2f}".format(clf.best_score_))
    # Prediction
    # y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
    return y_test_pred_proba, clf

# def tuning_and_evaluation(model,parameters,X_train,y_train,X_test,y_test):
#     clf = GridSearchCV(model, parameters, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
#     # Fit model
#     clf.fit(X_train, y_train)
#     # Best model after grid search
#     # print("Best parameters found: ", clf.best_params_)
#     # print("Best cross-validated score: {:.2f}".format(clf.best_score_))
#     # print('')
#     # Evaluation
#     y_test_pred = clf.predict(X_test)
#     y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
#     # print('Performance on Test Set')
#     auc, auc_ci_lower, auc_ci_upper = compute_auroc_ci(y_test.values, y_test_pred_proba)
#     # print(f"AUROC: {auc:.4f}")
#     # print(f"AUROC 95% CI: {auc:.4f}[{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
#     acc, acc_ci_lower, acc_ci_upper = compute_accuracy_ci(y_test.values, y_test_pred)
#     # print(f"Accuracy 95% CI: {acc:.4f}[{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]")
#     return auc,auc_ci_lower,auc_ci_upper,acc,acc_ci_lower,acc_ci_upper

"""
Save and load data
"""
# Save dataframe into npz format
def save_npz(df,path):
    # Convert DataFrame to NumPy arrays
    data = df.to_numpy()
    columns = df.columns.to_numpy()
    index = df.index.to_numpy()
    # Save to NPZ file
    np.savez(path, data=data, columns=columns, index=index)
    print(f'File saved to path {path}')
    
# Load npz file and reconstruct dataframe
def load_npz(path):
    # Load NPZ file
    npzfile = np.load(path, allow_pickle=True)
    # Extract arrays
    data = npzfile['data']
    columns = npzfile['columns']
    index = npzfile['index']
    # Reconstruct DataFrame
    df_loaded = pd.DataFrame(data, columns=columns, index=index)
    return df_loaded