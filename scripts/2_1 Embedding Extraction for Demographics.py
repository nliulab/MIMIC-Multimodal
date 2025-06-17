import os
import numpy as np
import pandas as pd
import pickle
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import argparse
from embedding_utils import *
import warnings
warnings.filterwarnings('ignore')

# args
parser = argparse.ArgumentParser()

## read and load files
parser.add_argument('--input-icu-path', type=str)
parser.add_argument('--input-metadata-path', type=str)
parser.add_argument('--output-path', type=str)

# variable list
parser.add_argument('--numerical-variables', nargs='+', default=['age'])
parser.add_argument('--categorical-variables', nargs='+', default=['gender', 'race', 'marital_status', 'language', 'insurance'])

args = parser.parse_args()

if __name__ == "__main__":
    
    # --input-icu-path './Data/processed_icu_24h.pkl' --input-metadata-path './Data/metadata_24h.csv' --output-path 'Embeddings/'
    
    print(args)
    input_icu_path = args.input_icu_path
    input_metadata_path = args.input_metadata_path
    output_path = args.output_path
    
    # Load data
    icus = pickle.load(open(input_icu_path,'rb'))
    icus_metadata = pd.read_csv(input_metadata_path,index_col=0)
    # print(icus_metadata.head(5))
    
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
    
    # Extract demographics information
    tabular_train = extract_tabular_information(icus_train,idx_train)
    tabular_val = extract_tabular_information(icus_val,idx_val)
    tabular_test = extract_tabular_information(icus_test,idx_test)
    
    # Standardization
    numerical_variables = args.numerical_variables
    categorical_variables = args.categorical_variables
    # Find least frequent categories for each categorical variable
    min_frequency = find_least_frequent_categories(tabular_train, categorical_variables)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age']),
            ('cat', OneHotEncoder(drop=min_frequency), categorical_variables)], # Drop categories with the least frequency
        remainder='passthrough'
        )
    # Transformation
    tabular_train_transformed = preprocessor.fit_transform(tabular_train)
    tabular_val_transformed = preprocessor.transform(tabular_val)
    tabular_test_transformed = preprocessor.transform(tabular_test)
    # Get feature names
    categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_variables)
    transformed_feature_names = numerical_variables + list(categorical_names)
    passthrough_feature_names = [i for i in tabular_train.columns if i not in numerical_variables and i not in categorical_variables]
    # Create dfs
    tabular_train_transformed = pd.DataFrame(tabular_train_transformed, columns=transformed_feature_names+passthrough_feature_names)
    tabular_val_transformed = pd.DataFrame(tabular_val_transformed, columns=transformed_feature_names+passthrough_feature_names)
    tabular_test_transformed = pd.DataFrame(tabular_test_transformed, columns=transformed_feature_names+passthrough_feature_names)
    # Reorder the columns
    tabular_train_transformed = tabular_train_transformed[passthrough_feature_names + transformed_feature_names]
    tabular_val_transformed = tabular_val_transformed[passthrough_feature_names + transformed_feature_names]
    tabular_test_transformed = tabular_test_transformed[passthrough_feature_names + transformed_feature_names]
    # Reindex
    tabular_train_transformed.index = tabular_train.index
    tabular_val_transformed.index = tabular_val.index
    tabular_test_transformed.index = tabular_test.index
    
    # Save embeddings
    save_npz(tabular_train_transformed, os.path.join(output_path,'tabular_embeddings_train.npz'))
    save_npz(tabular_val_transformed, os.path.join(output_path,'tabular_embeddings_val.npz'))
    save_npz(tabular_test_transformed, os.path.join(output_path,'tabular_embeddings_test.npz'))