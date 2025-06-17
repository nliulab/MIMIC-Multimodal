import os
import numpy as np
import pandas as pd
import pickle
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from momentfm import MOMENTPipeline
from embedding_utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

# args
parser = argparse.ArgumentParser()

## read and load files
parser.add_argument('--input-icu-path', type=str, default='Data/processed_icu_24h.pkl')
parser.add_argument('--input-metadata-path', type=str, default='Data/metadata_24h.csv')
parser.add_argument('--output-path', type=str, default='Updated Embeddings/')

## embedding techniques & outcome of interest
parser.add_argument('--emb-technique', type=str, help='fixed time interval/gru/moment',default='gru')
parser.add_argument('--outcome', type=str, default='los_binary')

## parameters
parser.add_argument('--total-length', type=int, default=24)
parser.add_argument('--bin-length', type=int, default=1)
parser.add_argument('--imputation', type=str, default='forward')

parser.add_argument('--hidden-size', type=int, default=64)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--dropout-rate', type=float, default=0.25)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-epochs', type=int, default=20)
# Best Params for mortality: {'hidden_size': 64, 'num_layers': 3, 'dropout_rate': 0.25, 'learning_rate': 0.001, 'batch_size': 32, 'num_epochs': 50}
# Best Params for LOS: {'hidden_size': 64, 'num_layers': 2, 'dropout_rate': 0.25, 'learning_rate': 0.001, 'batch_size': 64, 'num_epochs': 20}

parser.add_argument('--moment-batch-size', type=int, default=32)

args = parser.parse_args()

if __name__ == "__main__":
    # --input-icu-path './Data/processed_icu_24h.pkl' --input-metadata-path './Data/metadata_24h.csv' --output-path 'Embeddings/'
    # --emb-technique 'fixed_time_interval'
    print(args)
    input_icu_path = args.input_icu_path
    input_metadata_path = args.input_metadata_path
    output_path = args.output_path
    
    # Load data
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
    
    # Outlier removal
    vital_signs_valid_range = {'Temperature Fahrenheit': {'outlier_low': 14.2*9/5+32, 'valid_low': 26*9/5+32, 
                                                      'valid_high': 45*9/5+32, 'outlier_high':47*9/5+32},# Celsius to Fahrenheit
                           'Heart Rate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 350, 'outlier_high':390},
                           'Respiratory Rate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 300, 'outlier_high':330},
                           'O2 saturation pulseoxymetry': {'outlier_low': 0, 'valid_low': 0, 
                                                           'valid_high': 100, 'outlier_high':150},
                           'Non Invasive Blood Pressure systolic': {'outlier_low': 0, 'valid_low': 0, 
                                                                    'valid_high': 375, 'outlier_high':375},
                           'Non Invasive Blood Pressure diastolic': {'outlier_low': 0, 'valid_low': 0, 
                                                                     'valid_high': 375, 'outlier_high':375},
                           'Arterial Blood Pressure systolic': {'outlier_low': 0, 'valid_low': 0, 
                                                                    'valid_high': 375, 'outlier_high':375},
                           'Arterial Blood Pressure diastolic': {'outlier_low': 0, 'valid_low': 0, 
                                                                     'valid_high': 375, 'outlier_high':375}
                          }
    
    # Choose embedding technique
    emb_technique = args.emb_technique
    total_length = args.total_length
    imputation = args.imputation

    if emb_technique == "fixed_time_interval":
        bin_length = args.bin_length
        # Embeddings
        ts_emb = extract_fixed_interval_ts_embedding(icus,total_length,bin_length,imputation,vital_signs_valid_range)
        # Variables that are not measured for entire stay remain NaN. Impute with global median
        fixed_ti_embedding = data_imputation(ts_emb,'median')
        # print(fixed_ti_embedding.head(10))
        # Export
        save_npz(fixed_ti_embedding, os.path.join(output_path,f'{emb_technique}_embeddings.npz'))
    
    elif emb_technique == "gru":
        outcome = args.outcome
        # Pre-processing
        print('Data Preparation')
        global_median = calculate_global_median(icus_train,vital_signs_valid_range,total_length)
        # Train
        measurements_train,masks_train,time_intervals_train,indices_train,seq_train = extract_ts_information(icus_train,metadata_train,
                                                                                                             total_length,imputation,vital_signs_valid_range,global_median)
        outcomes_train = metadata_train.loc[indices_train,outcome]
        # Val
        measurements_val,masks_val,time_intervals_val,indices_val,seq_val = extract_ts_information(icus_val,metadata_val,
                                                                                                   total_length,imputation,vital_signs_valid_range,global_median)
        outcomes_val = metadata_val.loc[indices_val,outcome]
        # Test
        measurements_test,masks_test,time_intervals_test,indices_test,seq_test = extract_ts_information(icus_test,metadata_test,
                                                                                                        total_length,imputation,vital_signs_valid_range,global_median)
        outcomes_test = metadata_test.loc[indices_test,outcome]
        # Choose 99.9 percentile as pad length
        pad_len = int(np.percentile(seq_train, 99.9))
        # print(f"Padding length: {pad_len}")
        # Standardize & Add masking & Pad sequences into the same length
        # Normalizer
        scaler = StandardScaler()
        all_measurements_train = pd.concat(measurements_train)
        scaler.fit(all_measurements_train)
        ## train
        padded_X_train = scale_pad_stack(measurements_train,masks_train,time_intervals_train,scaler,pad_len)
        ## validation
        padded_X_val = scale_pad_stack(measurements_val,masks_val,time_intervals_val,scaler,pad_len)
        ## test
        padded_X_test = scale_pad_stack(measurements_test,masks_test,time_intervals_test,scaler,pad_len)
        # Convert outcome to ndarray
        y_train = np.array(outcomes_train)
        y_val = np.array(outcomes_val)
        y_test = np.array(outcomes_test)
        # Convert to tensor
        X_train_tensor = torch.tensor(padded_X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(padded_X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(padded_X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        # Create dataset
        train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor,y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor,y_test_tensor)
        # print(X_train_tensor.shape,y_train_tensor.shape)
        # print(X_val_tensor.shape,y_val_tensor.shape)
        # print(X_test_tensor.shape,y_test_tensor.shape)
    
        # Embedding extraction
        print('Embedding Extraction')
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_features = X_train_tensor.shape[2]
        seq_length = X_train_tensor.shape[1]
        params = {'hidden_size': args.hidden_size, 'num_layers': args.num_layers, 
              'dropout_rate': args.dropout_rate, 'learning_rate': args.learning_rate, 
              'batch_size': args.batch_size, 'num_epochs': args.num_epochs}
    
        # Initiate model
        model = GRUModel(num_features, params['hidden_size'], params['num_layers'], 1, seq_length, params['dropout_rate'])
        # Create dataloader
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        # Optimize
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        # Train the model
        # print('Model Training')
        trained_model = train_gru(model, train_loader, val_loader, criterion, optimizer, num_epochs=params['num_epochs'])
        # Recreate the train_loader and set shuffle=False
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
        # Extract features from the final dense layer
        embedding_train = extract_gru_ts_embedding(trained_model, train_loader, device)
        embedding_val= extract_gru_ts_embedding(trained_model, val_loader, device)
        embedding_test = extract_gru_ts_embedding(trained_model, test_loader, device)
        # Combine extracted embedding with indexes
        ## Train
        gru_embedding_train = pd.DataFrame(embedding_train)
        gru_embedding_train.index = indices_train
        gru_embedding_train.index.name = 'Index'
        # Validation
        gru_embedding_val = pd.DataFrame(embedding_val)
        gru_embedding_val.index = indices_val
        gru_embedding_val.index.name = 'Index'
        # Test
        gru_embedding_test = pd.DataFrame(embedding_test)
        gru_embedding_test.index = indices_test
        gru_embedding_test.index.name = 'Index'
    
        # Export
        save_npz(gru_embedding_train,os.path.join(output_path,f'{emb_technique}_{outcome}_embeddings_train.npz'))
        save_npz(gru_embedding_val,os.path.join(output_path,f'{emb_technique}_{outcome}_embeddings_val.npz'))
        save_npz(gru_embedding_test,os.path.join(output_path,f'{emb_technique}_{outcome}_embeddings_test.npz'))
    
    elif emb_technique == 'moment':
        print('Data Preparation')
        global_median = calculate_global_median(icus_train,vital_signs_valid_range,total_length)
        # Train
        measurements_train,masks_train,time_intervals_train,indices_train,seq_train = extract_ts_information(icus_train,metadata_train,
                                                                                                             total_length,imputation,vital_signs_valid_range,global_median)
        # Val
        measurements_val,masks_val,time_intervals_val,indices_val,seq_val = extract_ts_information(icus_val,metadata_val,
                                                                                                   total_length,imputation,vital_signs_valid_range,global_median)
        # Test
        measurements_test,masks_test,time_intervals_test,indices_test,seq_test = extract_ts_information(icus_test,metadata_test,
                                                                                                        total_length,imputation,vital_signs_valid_range,global_median)
        # Choose 99.9 percentile as pad length
        pad_len = int(np.percentile(seq_train, 99.9))
        # Pad
        padded_measurements_train,input_masks_train = pad_mask(measurements_train,pad_len)
        padded_measurements_val,input_masks_val = pad_mask(measurements_val,pad_len)
        padded_measurements_test,input_masks_test = pad_mask(measurements_test,pad_len)
        # Create dataset
        moment_batch_size = args.moment_batch_size
        train_dataset = TensorDataset(padded_measurements_train,input_masks_train)
        val_dataset = TensorDataset(padded_measurements_val,input_masks_val)
        test_dataset = TensorDataset(padded_measurements_test,input_masks_test)
        # Dataloader
        train_loader = DataLoader(train_dataset, moment_batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, moment_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, moment_batch_size, shuffle=False)
        
        print('Embedding Extraction')
        # Initiate model
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={"task_name": "embedding"},
            )
        model.init()
        # Extract embeddings
        moment_embedding_train = extract_moment_ts_embedding(model, train_loader,input_masks=True)
        moment_embedding_val = extract_moment_ts_embedding(model, val_loader,input_masks=True)
        moment_embedding_test = extract_moment_ts_embedding(model, test_loader,input_masks=True)
        # Combine extracted embedding with indexes
        # Train
        moment_embedding_train = pd.DataFrame(moment_embedding_train)
        moment_embedding_train.index = indices_train
        moment_embedding_train.index.name = 'Index'
        # Validation
        moment_embedding_val = pd.DataFrame(moment_embedding_val)
        moment_embedding_val.index = indices_val
        moment_embedding_val.index.name = 'Index'
        # Test
        moment_embedding_test = pd.DataFrame(moment_embedding_test)
        moment_embedding_test.index = indices_test
        moment_embedding_test.index.name = 'Index'
        
        # Export
        save_npz(moment_embedding_train,os.path.join(output_path,f'{emb_technique}_embeddings_train.npz'))
        save_npz(moment_embedding_val,os.path.join(output_path,f'{emb_technique}_embeddings_val.npz'))
        save_npz(moment_embedding_test,os.path.join(output_path,f'{emb_technique}_embeddings_test.npz'))