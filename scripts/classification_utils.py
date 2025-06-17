import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,recall_score,average_precision_score
from sklearn.model_selection import GridSearchCV
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

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

def tuning_and_evaluation(model,parameters,X_train,y_train,X_test,y_test):
    clf = GridSearchCV(model, parameters, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    # Fit model
    clf.fit(X_train, y_train)
    # Best model after grid search
    # print("Best parameters found: ", clf.best_params_)
    # print("Best cross-validated score: {:.2f}".format(clf.best_score_))
    # print('')
    # Evaluation
    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
    # print('Performance on Test Set')
    auc, auc_ci_lower, auc_ci_upper = compute_auroc_ci(y_test.values, y_test_pred_proba)
    # print(f"AUROC: {auc:.4f}")
    # print(f"AUROC 95% CI: {auc:.4f}[{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
    acc, acc_ci_lower, acc_ci_upper = compute_accuracy_ci(y_test.values, y_test_pred)
    # print(f"Accuracy 95% CI: {acc:.4f}[{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]")
    return auc,auc_ci_lower,auc_ci_upper,acc,acc_ci_lower,acc_ci_upper

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

"""
Evaluation --- Performance
"""
# Calculate confidence interval of AUROC
def compute_auroc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUROC) with a 95% Confidence Interval (CI).
    
    Parameters:
    - y_true: True binary labels.
    - y_pred: array-like of shape (n_samples,) Predicted probabilities.
    - n_bootstraps: int, default=1000 Number of bootstrap samples.
    - alpha: float, default=0.95 Confidence level for the confidence interval.
        
    Returns:
    - auc: float AUROC score.
    - ci_lower: float Lower bound of the confidence interval.
    - ci_upper: float Upper bound of the confidence interval.
    """
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for AUROC to be defined
            continue
        
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    # Compute the lower and upper bound of the confidence interval
    sorted_scores = np.sort(bootstrapped_scores)
    ci_lower = np.percentile(sorted_scores, (1.0 - alpha) / 2.0 * 100)
    ci_upper = np.percentile(sorted_scores, (1.0 + alpha) / 2.0 * 100)
    
    auc = roc_auc_score(y_true, y_pred)
    
    return auc, ci_lower, ci_upper

# Calculate confidence interval of accuracy
def compute_accuracy_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    # Compute the Accuracy with a 95% Confidence Interval (CI) using bootstrap.
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    bootstrapped_accuracies = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred_binary), len(y_pred_binary))
        
        score = accuracy_score(y_true[indices], y_pred_binary[indices])
        bootstrapped_accuracies.append(score)
    
    # Compute the lower and upper bound of the confidence interval
    sorted_accuracies = np.sort(bootstrapped_accuracies)
    ci_lower = np.percentile(sorted_accuracies, (1.0 - alpha) / 2.0 * 100)
    ci_upper = np.percentile(sorted_accuracies, (1.0 + alpha) / 2.0 * 100)
    
    acc = accuracy_score(y_true, y_pred_binary)
    
    return acc, ci_lower, ci_upper

def compute_tpr_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    # Compute the True Positive Rate (TPR) with a 95% Confidence Interval (CI) using bootstrap.
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    bootstrapped_tprs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred_binary), len(y_pred_binary))
        
        # Calculate TPR for the bootstrap sample
        score = recall_score(y_true[indices], y_pred_binary[indices])
        bootstrapped_tprs.append(score)
    
    # Compute the lower and upper bound of the confidence interval
    sorted_tprs = np.sort(bootstrapped_tprs)
    ci_lower = np.percentile(sorted_tprs, (1.0 - alpha) / 2.0 * 100)
    ci_upper = np.percentile(sorted_tprs, (1.0 + alpha) / 2.0 * 100)
    
    # Compute the TPR on the original data
    tpr = recall_score(y_true, y_pred_binary)
    
    return tpr, ci_lower, ci_upper

def compute_fpr_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    # Compute the False Positive Rate (FPR) with a 95% Confidence Interval (CI) using bootstrap.
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    bootstrapped_fprs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred_binary), len(y_pred_binary))
        
        # Calculate FPR for the bootstrap sample
        tn, fp, _, _ = confusion_matrix(y_true[indices], y_pred_binary[indices]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        bootstrapped_fprs.append(fpr)
    
    # Compute the lower and upper bound of the confidence interval
    sorted_fprs = np.sort(bootstrapped_fprs)
    ci_lower = np.percentile(sorted_fprs, (1.0 - alpha) / 2.0 * 100)
    ci_upper = np.percentile(sorted_fprs, (1.0 + alpha) / 2.0 * 100)
    
    # Compute the FPR on the original data
    tn, fp, _, _ = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return fpr, ci_lower, ci_upper

def compute_auprc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    # Compute the Area Under the Precision-Recall Curve (AUPRC) with a 95% Confidence Interval (CI).

    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for AUPRC to be defined
            continue
        
        score = average_precision_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    # Compute the lower and upper bound of the confidence interval
    sorted_scores = np.sort(bootstrapped_scores)
    ci_lower = np.percentile(sorted_scores, (1.0 - alpha) / 2.0 * 100)
    ci_upper = np.percentile(sorted_scores, (1.0 + alpha) / 2.0 * 100)
    
    auprc = average_precision_score(y_true, y_pred)
    
    return auprc, ci_lower, ci_upper

def perf_evaluation(outcome,ts_model,img_model,txt_model,path_dir):
    # Obtain result file path
    path = path_dir + f'{outcome}/{ts_model}_{img_model}_{txt_model}_results.pkl'
    models = f'{ts_model}+{img_model}+{txt_model}'
    
    # Retrieve results
    results = pickle.load(open(path, 'rb'))
    y_true = results['True label']
    
    # Initialize an empty list
    rows = []
    # Compute performance for each modality
    ## Baseline
    if 'Baseline' in list(results.keys()):
        y_pred_prob = results['Baseline']
        auc, auc_ci_lower, auc_ci_upper = compute_auroc_ci(y_true.values, y_pred_prob)
        acc, acc_ci_lower, acc_ci_upper = compute_accuracy_ci(y_true.values, y_pred_prob)
        tpr,tpr_ci_lower,tpr_ci_upper = compute_tpr_ci(y_true.values, y_pred_prob)
        fpr,fpr_ci_lower,fpr_ci_upper = compute_fpr_ci(y_true.values, y_pred_prob)
        auprc,auprc_ci_lower,auprc_ci_upper = compute_auprc_ci(y_true.values, y_pred_prob)
        rows.append((models, "Baseline", 
                     f'{auc:.4f}[{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]', 
                     f'{acc:.4f}[{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]',
                     f'{tpr:.4f}[{tpr_ci_lower:.4f}, {tpr_ci_upper:.4f}]',
                     f'{fpr:.4f}[{fpr_ci_lower:.4f}, {fpr_ci_upper:.4f}]',
                     f'{auprc:.4f}[{auprc_ci_lower:.4f}, {auprc_ci_upper:.4f}]'))

    ## Baseline+Image
    if 'Baseline+Image' in list(results.keys()):
        y_pred_prob = results['Baseline+Image']
        auc, auc_ci_lower, auc_ci_upper = compute_auroc_ci(y_true.values, y_pred_prob)
        acc, acc_ci_lower, acc_ci_upper = compute_accuracy_ci(y_true.values, y_pred_prob)
        tpr,tpr_ci_lower,tpr_ci_upper = compute_tpr_ci(y_true.values, y_pred_prob)
        fpr,fpr_ci_lower,fpr_ci_upper = compute_fpr_ci(y_true.values, y_pred_prob)
        auprc,auprc_ci_lower,auprc_ci_upper = compute_auprc_ci(y_true.values, y_pred_prob)
        rows.append((models, "Baseline+Image", 
                     f'{auc:.4f}[{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]', 
                     f'{acc:.4f}[{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]',
                     f'{tpr:.4f}[{tpr_ci_lower:.4f}, {tpr_ci_upper:.4f}]',
                     f'{fpr:.4f}[{fpr_ci_lower:.4f}, {fpr_ci_upper:.4f}]',
                     f'{auprc:.4f}[{auprc_ci_lower:.4f}, {auprc_ci_upper:.4f}]'))

    ## Baseline+Text
    if 'Baseline+Text' in list(results.keys()):
        y_pred_prob = results['Baseline+Text']
        auc, auc_ci_lower, auc_ci_upper = compute_auroc_ci(y_true.values, y_pred_prob)
        acc, acc_ci_lower, acc_ci_upper = compute_accuracy_ci(y_true.values, y_pred_prob)
        tpr,tpr_ci_lower,tpr_ci_upper = compute_tpr_ci(y_true.values, y_pred_prob)
        fpr,fpr_ci_lower,fpr_ci_upper = compute_fpr_ci(y_true.values, y_pred_prob)
        auprc,auprc_ci_lower,auprc_ci_upper = compute_auprc_ci(y_true.values, y_pred_prob)
        rows.append((models, "Baseline+Text", 
                     f'{auc:.4f}[{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]', 
                     f'{acc:.4f}[{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]',
                     f'{tpr:.4f}[{tpr_ci_lower:.4f}, {tpr_ci_upper:.4f}]',
                     f'{fpr:.4f}[{fpr_ci_lower:.4f}, {fpr_ci_upper:.4f}]',
                     f'{auprc:.4f}[{auprc_ci_lower:.4f}, {auprc_ci_upper:.4f}]'))

    ## Baseline+Image+Text
    if 'Baseline+Image+Text' in list(results.keys()):
        y_pred_prob = results['Baseline+Image+Text']
        auc, auc_ci_lower, auc_ci_upper = compute_auroc_ci(y_true.values, y_pred_prob)
        acc, acc_ci_lower, acc_ci_upper = compute_accuracy_ci(y_true.values, y_pred_prob)
        tpr,tpr_ci_lower,tpr_ci_upper = compute_tpr_ci(y_true.values, y_pred_prob)
        fpr,fpr_ci_lower,fpr_ci_upper = compute_fpr_ci(y_true.values, y_pred_prob)
        auprc,auprc_ci_lower,auprc_ci_upper = compute_auprc_ci(y_true.values, y_pred_prob)
        rows.append((models, "Baseline+Image+Text", 
                     f'{auc:.4f}[{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]', 
                     f'{acc:.4f}[{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]',
                     f'{tpr:.4f}[{tpr_ci_lower:.4f}, {tpr_ci_upper:.4f}]',
                     f'{fpr:.4f}[{fpr_ci_lower:.4f}, {fpr_ci_upper:.4f}]',
                     f'{auprc:.4f}[{auprc_ci_lower:.4f}, {auprc_ci_upper:.4f}]'))
    
    return rows

"""
Evaluation --- Fairness
"""
def group_age(age,cutoffs=[65, 85]):
    if age <= cutoffs[0]:
        return f"Age<={cutoffs[0]}"
    elif age <= cutoffs[1]:
        return f"{cutoffs[0]}<Age<={cutoffs[1]}"
    else:
        return f"Age>{cutoffs[1]}"
    
# Performance metrics for different subgroups of sensitive features
def fair_metrics(y_true,y_pred_proba,sensitive_features):
    results = []

    # Loop through each unique value in sensitive_features
    for feature in sensitive_features.unique():
        # Filter the data for the current group
        mask = sensitive_features == feature
        y_true_group = y_true[mask]
        y_pred_proba_group = y_pred_proba[mask]
        # Generate y_pred based on y_pred_proba > 0.5
        y_pred_group = (y_pred_proba_group >= 0.5).astype(int)

        # Calculate AUROC for the current group
        auroc = roc_auc_score(y_true_group, y_pred_proba_group)
        # Calculate accuracy for the current group
        accuracy = accuracy_score(y_true_group, y_pred_group)
        # Calculate TPR and FPR for the current group
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        tpr = tp / (tp + fn)  # True Positive Rate
        fpr = fp / (fp + tn)  # False Positive Rate
        # Calculate selection rate
        sr = sum(y_pred_group)/len(y_pred_group)
        
        # Append the result as a row
        results.append([feature, auroc, accuracy, tpr, fpr,sr])

    # Convert the results into a DataFrame and set the index
    df = pd.DataFrame(results, columns=[sensitive_features.name, 'AUROC', 'Accuracy', 'True Positive Rate', 
                                        'False Positive Rate','Selection Rate'])
    df.set_index(sensitive_features.name, inplace=True)

    return df

# Integrated fairness evaluation
def fair_evaluation(y_true,y_pred_proba,sensitive_feature_list,title):
    perf_dic = {}
    fair_measures_dic = {}
    for sensitive_feature in sensitive_feature_list:
        name = sensitive_feature.name
        # print(name)
        
        # Compare auroc,accuracy,tpr, fpr and selection rate between different groups
        perf = fair_metrics(y_true,y_pred_proba,sensitive_feature)
        # Add title
        perf.columns = pd.MultiIndex.from_tuples([(title, col) for col in perf.columns])
        perf_dic[name] = perf.T
        
        # Calculate y_pred based on y_pred_proba >= 0.5
        y_pred = (y_pred_proba >= 0.5).astype(int)
        # Calculate demogrphic parity and equalized odds
        fair_measures = pd.DataFrame(index=[title],columns=['Demographic Parity','Equalized Odds'])
        fair_measures['Demographic Parity'] = demographic_parity_ratio(y_true,y_pred,sensitive_features=sensitive_feature)
        fair_measures['Equalized Odds'] = equalized_odds_ratio(y_true,y_pred,sensitive_features=sensitive_feature)
        fair_measures.columns = pd.MultiIndex.from_tuples([(name.capitalize(), col) for col in fair_measures.columns])
        fair_measures_dic[name] = fair_measures
        
    return perf_dic,fair_measures_dic