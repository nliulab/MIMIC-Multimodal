import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

"""
General
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
Embedding Extraction for Demographics
"""

# Regroup race pre-defined groups
def regroup_race(race, rgs=["WHITE", "ASIAN", "HISPANIC", "BLACK"]):
    """
    Group race into one of the four main categories: ['WHITE', 'ASIAN', 'HISPANIC', 'BLACK'].
    If the race does not contain any of these categories, classify it as 'OTHERS'.
    """
    if np.sum([1 if g in race else 0 for g in rgs]) == 0:
        return "OTHER"
    else:
        for g in rgs:
            if g in race:
                return g

# Get tabular information
def extract_tabular_information(icus,indexes):
    tabular = pd.DataFrame()
    with tqdm(total=len(indexes)) as pbar:
        for i in range(len(indexes)):
            icu = icus[i]
            tab = icu.tabular
            idx = indexes[i]
            # Regroup race
            if 'race' in tab.columns:
                tab['race'] = tab['race'].apply(regroup_race)
            # Re-index
            tab.index = [idx]
            tab.index.name = 'Index'
            # Missing data imputation
            tab['marital_status'] = tab['marital_status'].fillna('UNKNOWN')
            # Concatenate
            tabular = pd.concat([tabular,tab])
            # Update
            pbar.update(1)
    
    return tabular

# Find least frequent categories for each categorical variable
def find_least_frequent_categories(df, categorical_variables):
    least_frequent_categories = []
    for col in categorical_variables:
        least_frequent_category = df[col].value_counts().idxmin()
        least_frequent_categories.append(least_frequent_category)
    return least_frequent_categories

"""
Embedding Extraction for Time-series --- Fixed Time Interval Transformation
"""
# Remove outliers
def outlier_imputation(x,column_range):
        if x < column_range['outlier_low'] or x > column_range['outlier_high']:
            # set as missing
            return np.nan
        elif x < column_range['valid_low']:
            # impute with nearest valid value
            return column_range['valid_low']
        elif x > column_range['valid_high']:
            # impute with nearest valid value
            return column_range['valid_high']
        else:
            return x
        
def outlier_removal(df,valid_range):
    for key in valid_range.keys():
        # print(key)
        column_range = valid_range[key]
        df[key] = df[key].apply(lambda x:outlier_imputation(x,column_range))
    return df

# Impute missing values
def data_imputation(df,imputation_method):
    # Inputs:
    #   df -> input dataframe
    #   imputation_method -> methods for data impuation, including forward, mean and zero imputation
    #   In forward imputation, if the first observation is missing then apply backward imputation afterwards
    
    # Outputs:
    #   imputed_df -> imputed dataframe
    imputed_df = df.copy()
    if imputation_method == 'forward':
        imputed_df = imputed_df.fillna(method='ffill')
        imputed_df = imputed_df.fillna(method='bfill') # backward fill for any remaining missing values
    elif imputation_method == 'mean':
        imputed_df = imputed_df.fillna(imputed_df.mean())
    elif imputation_method == 'zero':
        imputed_df = imputed_df.fillna(0)
    elif imputation_method == 'median':
        imputed_df = imputed_df.fillna(imputed_df.median())
    return imputed_df

# Integrate irregular sampled time-series data into fixed time intervals
def fixed_time_aggregation(df,total_length,bin_length):
    # Inputs:
    #   df -> input dataframe
    #   total_length -> total length of time over which data is collected
    #   bin_length -> length of of each time bin for data aggregation
    
    # Outputs:
    #   filtered_df -> output dataframe
    if total_length % bin_length != 0:
        raise ValueError(f"The bin length {bin_length} does not divide evenly into the total time length {total_length}.")

    filtered_df = df[df['frac_charttime'] < total_length]
    filtered_df['time_bin'] = (filtered_df['frac_charttime'] // bin_length).astype(int)
    filtered_df.drop(columns=['hadm_id','charttime','frac_charttime'],inplace=True)
    aggregated_df = filtered_df.groupby('time_bin').mean()
    # Reindex to include all time bins
    n_bins = total_length // bin_length
    aggregated_df = aggregated_df.reindex(range(0, n_bins), fill_value=np.nan)
    aggregated_df.reset_index(inplace=True)
    return aggregated_df

# Reshape dataframe
# e.g. df with shape (24,2) has 'time' and 'Heart Rate' variables collected at 24 time points.
# With this function, we can convert it into df with shape (1,24) and columns 'Heart Rate_0', ...,'Heart Rate_23'
def flatten(df):
    # orginial df (# time bins, # variables) -> flattened df (1, # time bins * # variables)
    var = df.columns[2:]
    variable_names = [f'{name}_{num}' for name in var for num in df['time_bin']]
    flattened_df = df[var].to_numpy().flatten(order='F')
    flattened_df = pd.DataFrame([flattened_df],columns=variable_names,index=df['stay_id'].unique())
    flattened_df.index.name = 'stay_id'
    return flattened_df

def extract_fixed_interval_ts_embedding(processed_ICUs,total_length,bin_length,imputation_method,valid_range):
    # Extract information for patient
    nfiles = len(processed_ICUs)
    with tqdm(total = nfiles) as pbar:
        embeddings = pd.DataFrame()
        for i in range(nfiles):
            processed_icu = processed_ICUs[i]
            df = processed_icu.time_series
            # Check if time-series data is missing
            # If missing, the embedding will be NaN
            if df.empty:
              missing_df = pd.DataFrame(np.nan,index=[i],columns=embeddings.columns)
              missing_df['stay_id'] = processed_icu.tabular['stay_id'].item()
              missing_df.index.name = 'Index'
              embeddings = pd.concat([embeddings,missing_df])
              pbar.update(1)
              continue
            # # Converting <NA> to NaN for all columns in the DataFrame
            # df = df.fillna(np.nan)
            # Remove outliers
            cleaned_df = outlier_removal(df,valid_range)
            # Aggregate the data to get a fixed length input
            filtered_df = fixed_time_aggregation(cleaned_df,total_length,bin_length)
            filtered_df['stay_id'] = filtered_df['stay_id'].fillna(method='ffill')
            # display(filtered_df)
            # Missing value imputation
            # Here, we impute missing data at ICU stay-level, thus variables that are not measured for entire stay remain NaN
            imputed_df = data_imputation(filtered_df,imputation_method)
            # display(imputed_df)
            # Flatten the dataframe into one row
            flattened_df = flatten(imputed_df)
            # Reindex the dataframe
            flattened_df = flattened_df.reset_index()
            flattened_df.index = [i]
            flattened_df.index.name = 'Index'
            # Concatenate by rows
            embeddings = pd.concat([embeddings,flattened_df])
            # Update process bar
            pbar.update(1)
    return embeddings

"""
Embedding Extraction for Time-series --- GRU
"""
## Pre-processing

def calculate_global_median(processed_ICUs,valid_range,total_length):
    nfiles = len(processed_ICUs)
    # print('Calculate global median')
    with tqdm(total=nfiles) as pbar:
        # Collect all original measurements to calculate global median
        all_measurements = []
        for i in range(nfiles):
            # Input
            processed_icu = processed_ICUs[i]
            df = processed_icu.time_series
            if df.empty:
                pbar.update(1)
                continue
            # Remove outliers
            cleaned_df = outlier_removal(df,valid_range)
            # Filter by time range
            measurement = cleaned_df[cleaned_df['frac_charttime'] < total_length].loc[:,"Heart Rate":"Temperature Fahrenheit"]
            all_measurements.append(measurement)
            pbar.update(1)
        combined_measurements = pd.concat(all_measurements)
        global_median = combined_measurements.median()
        # print(global_median)
        return global_median
    
def ts_preprocessing(df,total_length,imputation_method,valid_range,global_median):
    # Remove outliers
    cleaned_df = outlier_removal(df,valid_range)
    # Filter data by ICU stay length
    filtered_df = cleaned_df[cleaned_df['frac_charttime'] < total_length]
    # Get measurement and time interval
    measurement = filtered_df.loc[:,"Heart Rate":"Temperature Fahrenheit"]
    time_interval = filtered_df['frac_charttime']
    # Add masking before imputation
    mask = ~measurement.isna()  # Mask indicating positions with actual data
    mask = mask.astype(int)
    # Missing value imputation
    imputed_measurement = data_imputation(measurement,imputation_method)
    # Columns with all values missing remain NaN after imputation
    # In this case, impute with global median
    imputed_measurement = imputed_measurement.fillna(global_median)
    return imputed_measurement,mask,time_interval

def extract_ts_information(processed_ICUs,metadata,total_length,imputation_method,valid_range,global_median):
    nfiles = len(processed_ICUs)
    # print('Information Extraction')
    with tqdm(total = nfiles) as pbar:
        measurements = []
        masks = []
        time_intervals = []
        seq_length_list = []
        indexes = []
        for i in range(nfiles):
            # Input
            processed_icu = processed_ICUs[i]
            df = processed_icu.time_series
            # Check if time-series data is missing
            if df.empty:
              pbar.update(1)
              continue
            # Get index
            stay_id = df['stay_id'].unique().item()
            idx = metadata[metadata['stay_id']==stay_id].index.item()
            if idx != metadata.index[i]:
                 raise ValueError("Index does not match")
            indexes.append(idx)
            # Extract information
            measurement,mask,time_interval = ts_preprocessing(df,total_length,imputation_method,valid_range,global_median)
            # Number of time points
            seq_length_list.append(measurement.shape[0])
            # Append
            measurements.append(measurement)
            masks.append(mask)
            time_intervals.append(time_interval)
            # Update progress
            pbar.update(1)
        # print('max length:',max(seq_length_list))
        # print('99.9 percentile of length:',np.percentile(seq_length_list, 99.9))
        # print('average length:',sum(seq_length_list)/nfiles)
    return measurements,masks,time_intervals,indexes,seq_length_list

# For GRU model, combine measurements,masks and time intervals after standardization and zero padding
def scale_pad_stack(measurements,masks,time_intervals,scaler,pad_length):
    padded_X = []
    for i in range(len(measurements)):
        df = measurements[i]
        mask = masks[i]
        time_interval = time_intervals[i].values.reshape(-1, 1)
        # Normalization
        scaled_df = scaler.transform(df)
        # Concatenate df with mask
        concat_df = np.concatenate([scaled_df,mask.values,time_interval],axis=1)
        # Padding
        padded_array = np.zeros((pad_length, concat_df.shape[1]))
        if concat_df.shape[0]<=pad_length:
            padded_array[:concat_df.shape[0], :] = concat_df
        else:
            padded_array = concat_df[:pad_length,:]
        padded_X.append(padded_array)
    padded_X = np.stack(padded_X)
    return padded_X

## Train model and extract embeddings
# GRU Model
class GRUModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,seq_length,dropout_rate):
        super(GRUModel,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        # GRU layer
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout_rate if num_layers > 1 else 0)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size,output_size)
        # Projection layer: From hidden_size to 1024
        self.projection = nn.Linear(hidden_size, 1024)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Sigmoid layer for probability output
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size).to(X.device)
        out, _ = self.gru(X, h0)
        # Use only the last time step's output
        out = out[:,-1,:]
        # Apply dropout
        out = self.dropout(out)
        out  =self.fc(out)
        probabilities = self.sigmoid(out)
        return probabilities
    
    def get_embeddings(self,X):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size).to(X.device)
        out, _ = self.gru(X, h0)
        # Use only the last time step's output
        out = out[:,-1,:]
        # Apply dropout
        out = self.dropout(out)
        # Project to 1024 dimensions
        projected_embedding = self.projection(out)
        return projected_embedding
    
# Define the training function with early stopping
def train_gru(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=3, max_grad_norm=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        if early_stop:
            print(f'Early stopping at epoch {epoch}')
            break
        
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        # Validate the model
        model.eval()
        val_loss = 0.0
        try:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
        
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
            # Check for NaN in val_loss
            if torch.isnan(torch.tensor(val_loss)):
                print(f'Epoch {epoch+1}, Val Loss is NaN, stopping training')
                early_stop = True
                break
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    early_stop = True
        
        except Exception as e:
            print(f"Exception encountered during validation at epoch {epoch+1}. Exception: {e}")
            early_stop = True
            break
    if early_stop:
        model.load_state_dict(best_model)
        
    return model

# Extract embedding
def extract_gru_ts_embedding(model, data_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            extracted_features = model.get_embeddings(inputs)
            features.append(extracted_features.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

"""
Embedding Extraction for Time-series --- Moment
"""

# For Moment, pad measurements into the same length and create masks
def pad_mask(measurements,pad_length):
    padded_measurements = []
    input_masks = []
    for m in measurements:
        length = m.shape[0]
        padded_m = np.full((pad_length, m.shape[1]),0)
        
        if length < pad_length:
            padded_m[:length, :] = m
            mask = np.concatenate([np.ones(length), np.zeros(pad_length-length)])
        else:
            padded_m = m.values[:pad_length,:]
            mask = np.ones(pad_length)
        
        padded_measurements.append(padded_m)
        input_masks.append(mask)
    # Stack
    padded_measurements = np.stack(padded_measurements)
    input_masks = np.stack(input_masks)
    # Convert to tensors
    padded_measurements = torch.tensor(padded_measurements, dtype=torch.float32)
    input_masks = torch.tensor(input_masks, dtype=torch.float32)
    # Permute the dimensions to (batch_size, channels, length)
    padded_measurements = padded_measurements.permute(0, 2, 1)
    
    return padded_measurements, input_masks

def extract_moment_ts_embedding(model,data_loader,input_masks=True):
    model.eval()
    all_embeddings = []
    i = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for batch in data_loader:
                inputs, masks = batch
                if input_masks:
                    output = model(inputs, input_mask=masks)
                else:
                    output = model(inputs)
                embedding = output.embeddings
                # if torch.isnan(embedding).any():
                #     print(f'Embeddings of batch {i} contain NaN values')
                all_embeddings.append(embedding.detach().cpu().numpy())
                i += 1
                pbar.update(1)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

"""
Embedding Extraction for Images --- Swin Transformer
"""
class SwinFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(SwinFeatureExtractor, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass through the model's feature extractor
        outputs = self.model.swin(x)  # Access the internal Swin module
        # Access the last hidden state
        features = outputs[-1]  
        return features

"""
Embedding Extraction for Clinical Notes
"""    
def combine_text(text):
    # Combine the reports with [SEP] token
    combined_text = " [SEP] ".join(text)
    return combined_text

def extract_radbert_text_embedding(text, model, tokenizer):
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=512
    )
    model.eval()
    with torch.no_grad():
        output = model(**encoded)
        embedding = output.last_hidden_state[:, 0, :]

    return embedding
