import os
import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import dask.dataframe as dd
from tqdm import tqdm
import re
import ast

"""
Master Dataset Generation
"""

## MIMICIV PATIENT CLASS STRUCTURE
class Patient_ICU(object):
    def __init__(self, core, admissions, patients, transfers, diagnoses_icd, procedures_icd, drgcodes,
                 services, labevents, hcpcsevents, microbiologyevents, emar, poe, prescriptions, 
                 icustays, procedureevents, outputevents, inputevents, datetimeevents, chartevents,ingredientevents,
                 cxr_split, cxr_metadata, cxr_chexpert, cxr_negbio, cxr_image_path, cxr_text_path, dsnotes, radnotes):
        
        ## CORE
        self.core = core
        
        ## HOSP
        self.admissions = admissions
        self.patients = patients
        self.transfers = transfers
        self.diagnoses_icd = diagnoses_icd
        self.procedures_icd = procedures_icd
        self.drgcodes = drgcodes
        self.services = services
        self.labevents = labevents
        self.hcpcsevents = hcpcsevents
        self.microbiologyevents = microbiologyevents
        self.emar = emar
        self.poe = poe
        self.prescriptions = prescriptions
        
        ## ICU
        self.icustays = icustays
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents
        self.ingredientevents = ingredientevents
        
        ## CXR
        # JPG
        self.cxr_split = cxr_split
        self.cxr_metadata = cxr_metadata
        self.cxr_chexpert = cxr_chexpert
        self.cxr_negbio = cxr_negbio
        # DICOM
        self.cxr_image_path = cxr_image_path
        self.cxr_text_path = cxr_text_path
        
        ## NOTES
        self.dsnotes = dsnotes
        self.radnotes = radnotes
        
## Read files in one folder into dask dataframes
def read_folder(dfs, folder_path):
    print(f'Read files in folder {folder_path}')
    file_list = os.listdir(folder_path)
    with tqdm(total=len(file_list)) as pbar:
        for file_name in file_list:
            # print(file_name)
            # Read the file when it ends with csv.gz format and is not already included
            if file_name.endswith('.csv.gz') and file_name[:-7] not in dfs.keys():
                # extract the name before ".csv.gz"
                name = file_name[:-7]  # removing ".csv.gz" (7 characters) from the file name
                file_path = os.path.join(folder_path, file_name)
                # read file
                ddf = dd.read_csv(file_path, compression='gzip', assume_missing=True, blocksize=None)
                # check if ddf can be successfully computed
                checked_ddf = check_ddf(ddf,file_path)
                # store the dataframe in a dictionary
                dfs[name] = checked_ddf
                # update progress
                pbar.update(1)
            else:
                # update progress
                pbar.update(1)
                continue
    return dfs

# Check if dask dataframe can be processed successfully
def check_ddf(ddf, path):
    # Errors may be raised when computing dask dataframes
    # Usually this is due to dask's dtype inference failing, and *may* be fixed by specifying dtypes manually
    try:
        ddf.head(5)
        return ddf
    except Exception as e:
        # Convert error message to string
        error_info = str(e)
        # Regular expression to capture the dtype suggestion
        pattern = r"(dtype=\{.*?\})"
        match = re.findall(pattern, error_info, re.DOTALL)
        dtype_dict = ast.literal_eval(match[0].split('=')[1].strip())
        # Read the dask dataframe with specified dtypes
        new_ddf = dd.read_csv(path, compression='gzip',assume_missing=True,dtype=dtype_dict)
        return new_ddf
    
# Convert time-related variables to datetime type
def convert_datetime(dfs):
    name_list = list(dfs.keys())
    for name in name_list:
        # print("{:-^50s}".format(name))
        df = dfs[name]
        for i in df.columns:
            j = i.lower()
            time_words = ["time","date","dod"]
            if any(word in j for word in time_words):
                # print(i)
                df[i] = dd.to_datetime(df[i],errors='coerce')
    return dfs


"""
Data Processing Pipeline
"""

## Processed Patient ICU Stay Structure
class Processed_ICU(object):
    def __init__(self,tabular,time_series,images,notes):
        self.tabular = tabular
        self.time_series = time_series
        self.images = images
        self.notes = notes

## Load Patient_ICU objects
def load_patient_ICU_objects(folder_path,file_list):
    # Load extracted patient_ICU files
    patient_icus = []
    nfiles = len(file_list)
    with tqdm(total = nfiles) as pbar:
        for file_name in file_list:
            if file_name.endswith('.pkl'):
                file_path = os.path.join(folder_path, file_name)
                patient_icu = pickle.load(open(file_path,'rb'))
                patient_icus.append(patient_icu)    
            # Update process bar
            pbar.update(1)
    return patient_icus

## Data pre-processing
def process_patient_ICU(patient_ICU, start_diff, end_diff, tabular_variables, vital_signs_variables, 
                        sorted, ascending):
        
    # Convert tables to pandas dataframes
    patient_ICU.patients = patient_ICU.patients.compute()
    
    ## Extract patient demographics
    tab = patient_ICU.admissions.merge(patient_ICU.patients,how='left',on='subject_id')
    tab = tab.merge(patient_ICU.icustays,how='left')
    tab['age'] = tab['anchor_age'] + (tab['admittime'].dt.year-tab['anchor_year'])
    tabular = tab[tabular_variables]
    metadata = tab[['subject_id','hadm_id','stay_id','admission_type', 
                    'admission_location', 'discharge_location', 'first_careunit', 'last_careunit',
                    'admittime','dischtime','edregtime','edouttime', 'intime', 'outtime',
                    'los','hospital_expire_flag','deathtime','dod']]
    
    ## Identify time bounds
    earliest_intime = tab[['admittime','edregtime','intime']].min(axis=1)
    icu_outtime = tab['outtime']
    # start time
    if start_diff is None:
        start_time = earliest_intime.iloc[0]
    else:
        if start_diff >=0:
            start_time = tab['intime'] + timedelta(hours=start_diff)
        else:
            start_time = tab['intime'] - timedelta(hours=abs(start_diff))
        start_time = max(start_time.iloc[0],earliest_intime.iloc[0])
    # end time
    if end_diff is None:
        end_time = icu_outtime.iloc[0]
    else:
        end_time = tab['intime'] + timedelta(hours=end_diff)
        end_time = min(end_time.iloc[0],icu_outtime.iloc[0])
        
    ## Extract time-series
    # time_series = {}
    # # lab
    # if lab_variables is not None:
    #     lab = patient_ICU.labevents[patient_ICU.labevents['label'].isin(lab_variables)]
    #     lab = lab[(lab['charttime']>start_time)&(lab['charttime']<end_time)]
    #     lab = lab.sort_values(by='charttime',ascending=ascending).reset_index(drop=True)
    #     time_series['lab'] = lab
    
    # chart
    if vital_signs_variables is not None:
        chart = patient_ICU.chartevents[patient_ICU.chartevents['label'].isin(vital_signs_variables)]
        chart = chart[(chart['charttime']>start_time)&(chart['charttime']<end_time)]
        chart = chart.sort_values(by='charttime',ascending=ascending).reset_index(drop=True)
        time_series = chart
    
    # procedure
    # if time_series_variables['procedure'] is not None:
    #     procedure = patient_ICU.procedureevents[patient_ICU.procedureevents['label'].isin(time_series_variables['procedure'])]
    #     procedure = procedure[(procedure['charttime']>start_time)&(procedure['charttime']<end_time)]
    #     procedure = procedure.sort_values(by='charttime',ascending=ascending).reset_index(drop=True)
    #     time_series['procedure'] = procedure
    
    ## Extract CXR
    images = {}
    # metadata
    cxr_metadata = patient_ICU.cxr_metadata
    cxr_metadata = cxr_metadata[(cxr_metadata['StudyDatetime']>start_time)&(cxr_metadata['StudyDatetime']<end_time)]
    if sorted:
        cxr_metadata = cxr_metadata.sort_values(by='StudyDatetime',ascending=ascending)
    cxr_metadata.reset_index(drop=True,inplace=True)
    images['metadata'] = cxr_metadata
    # path
    dicom_id_list = cxr_metadata['dicom_id'].unique()
    study_id_list = cxr_metadata['study_id'].unique()
    # CXR images
    image_path = patient_ICU.cxr_image_path
    image_path = image_path[image_path['dicom_id'].isin(dicom_id_list)]
    images['image_path'] = image_path
    # CXR reports
    text_path = patient_ICU.cxr_text_path.compute()
    text_path = text_path[text_path['study_id'].isin(study_id_list)]
    images['text_path'] = text_path
    
    ## Notes
    notes = {}
    # discharge summary
    dsnotes = patient_ICU.dsnotes
    notes['discharge']  = dsnotes
    # radiology reports
    radnotes = patient_ICU.radnotes
    radnotes = radnotes.drop(columns='subject_id_y')
    radnotes = radnotes.rename(columns={"subject_id_x": "subject_id"})
    radnotes = radnotes[(radnotes['charttime']>start_time)&(radnotes['charttime']<end_time)]
    if sorted:
        radnotes = radnotes.sort_values(by='charttime',ascending=ascending)
    notes['radiology'] = radnotes
    
    processed_ICU = Processed_ICU(tabular,time_series,images,notes)

    return processed_ICU, metadata

## Cohort selection based on inclusion criteria
def cohort_selection(processed_ICU,metadata,age_lower,age_upper,drop_missing_modalities,los_lower):
    # Filter by age
    age = processed_ICU.tabular['age'].item()
    age_inclusion = True if age >= age_lower and age <= age_upper else False
    # Filter by missing modality
    # Criteria for missing modalities
    drop_missing_ts, drop_missing_img, drop_missing_text = drop_missing_modalities
    ## Time-series
    missing_ts = processed_ICU.time_series.empty
    ts_inclusion = False if missing_ts and drop_missing_ts else True
    ## Image
    missing_img = processed_ICU.images['metadata'].empty
    img_inclusion = False if missing_img and drop_missing_img else True
    ## Text
    missing_text = True if processed_ICU.notes['discharge'].empty and processed_ICU.notes['radiology'].empty else False
    text_inclusion = False if missing_text and drop_missing_text else True
    ## Inclusion decision on modality
    modality_inclusion = True if ts_inclusion and img_inclusion and text_inclusion else False
    # Filter by length of stay
    los = metadata['los'].item()
    los_inclusion = True if los >= los_lower else False
    # Inclusion
    inclusion = True if age_inclusion and modality_inclusion and los_inclusion else False
    return inclusion#,missing_ts,missing_img,missing_text

## Reshape time series data
def time_series_reshaping(processed_ICU, vital_signs_variables, ascending):
    # vital signs
    if not processed_ICU.time_series.empty:
        # Reshape dataframe
        df = processed_ICU.time_series.copy()
        df_filtered = df[['hadm_id','stay_id','charttime','label','valuenum']]
        df_pivot = df_filtered.pivot(index=['hadm_id','stay_id','charttime'],columns='label',values='valuenum')
        df_pivot = df_pivot.sort_values(by='charttime',ascending=ascending)
        for var in vital_signs_variables:
            if var not in df_pivot.columns:
                df_pivot[var] = np.nan
        df_pivot = df_pivot.loc[:,vital_signs_variables]
        df_pivot.reset_index(inplace=True)
        # Add fractional charttime
        t0 = min(df_pivot['charttime'])
        frac_time = []
        for _, content in df_pivot.iterrows():
            ## Calculate time difference between current charttime and first charttime in hours
            time_diff = (content['charttime'] - t0).total_seconds() / 3600 
            frac_time.append(round(time_diff,2))
        df_pivot['frac_charttime'] = frac_time
        # # Drop rows with all vital signs values zero
        # df_pivot = df_pivot.dropna(subset=var, how='all')
        processed_ICU.time_series = df_pivot
    return processed_ICU

# Add outcome information in metadata
def extract_outcome(icus_metadata,los_range,readmission_range):
    # Convert time-related variables in metadata to datetime
    icus_metadata['admittime'] = pd.to_datetime(icus_metadata['admittime'])
    icus_metadata['dischtime'] = pd.to_datetime(icus_metadata['dischtime'])
    icus_metadata['edregtime'] = pd.to_datetime(icus_metadata['edregtime'])
    icus_metadata['edouttime'] = pd.to_datetime(icus_metadata['edouttime'])
    icus_metadata['intime'] = pd.to_datetime(icus_metadata['intime'])
    icus_metadata['outtime'] = pd.to_datetime(icus_metadata['outtime'])
    icus_metadata['deathtime'] = pd.to_datetime(icus_metadata['deathtime'])
    icus_metadata['dod'] = pd.to_datetime(icus_metadata['dod'])
         
    # Add in-ICU mortality
    icu_expire_flag = []
    for _,data in icus_metadata.iterrows():
        if data['hospital_expire_flag'] == 0:
            icu_expire_flag.append(0)
        elif pd.notna(data['deathtime']):
            expire = 1 if data['deathtime']>data['intime'] and data['deathtime']<data['outtime'] else 0
            icu_expire_flag.append(expire)
        else:
            expire = 1 if data['dod']>data['intime'] and data['dod']<data['outtime'] else 0
            icu_expire_flag.append(expire)
    icus_metadata['icu_expire_flag'] = icu_expire_flag
    
    # Add los>los_range
    los_binary = [1 if los > los_range else 0 for los in icus_metadata['los']]
    icus_metadata['los_binary'] = los_binary
    
    # Add re-admission
    readmission = []
    ## Iterate over each ICU stay
    for _, row in icus_metadata.iterrows():
        re_adm = 0
        subject_id = row['subject_id']
        intime = row['intime']
        outtime = row['outtime']
        # Find all stays for the same patient after the current stay
        subsequent_stays = icus_metadata[(icus_metadata['subject_id'] == subject_id) & (icus_metadata['intime'] > intime)]
        # Check if any subsequent stay is within 30 days
        if not subsequent_stays.empty:
            if any((subsequent_stays['intime'] - outtime) <= pd.Timedelta(days=readmission_range)):
                re_adm = 1
        readmission.append(re_adm)
    icus_metadata['readmission'] = readmission
    
    return icus_metadata

## Batch processing
def process_all(patient_ICUs,age_lower,age_upper,drop_missing_modalities,los_lower,
                start_diff,end_diff,tabular_variables,vital_signs_variables,sorted,ascending,
                los_range,readmission_range):
    '''
    Inputs:
      patient_ICUs -> list of extracted information for each single ICU stay
      age_lower -> Lower bound of patient age
      age_upper -> Upper bound of patient age
      drop_missing_modalities -> Drop objects with modalities missing (a list as input for time-series,image and text)
      los_lower -> Lower bound of patient length of stay
      start_diff -> Time difference between the start of information collection period and ICU admission time(positive/negative)
      end_diff -> Time difference between the end of information collection period and ICU admission time(positive only)
      tabular_variables -> List of tabular variables included
      vital_signs_variables -> List of vital signs variables included
      los_range -> Convert length of stay to binary outcome by a customized time range(days)
      readmission_range -> Time range for patient re-admission in days
    
    Outputs:
      processed_ICUs -> list of processed information for each single ICU stay
      patients_metadata -> metadata of processed ICU stays
    '''
    # Extract information for patient
    nfiles = len(patient_ICUs)
    with tqdm(total = nfiles) as pbar:
        #Iterate through all patients
        processed_ICUs = []
        patients_metadata = pd.DataFrame()
        for i in range(nfiles):
            patient_ICU = patient_ICUs[i]
            processed_ICU, metadata = process_patient_ICU(patient_ICU,start_diff,end_diff,tabular_variables,
                                                          vital_signs_variables,sorted,ascending)
            inclusion = cohort_selection(processed_ICU,metadata,age_lower,age_upper,drop_missing_modalities,los_lower)
            if not inclusion:
                # Update process bar
                pbar.update(1)
                continue
            # Reshape vital signs table
            processed_ICU = time_series_reshaping(processed_ICU,vital_signs_variables,ascending)
            # Append
            processed_ICUs.append(processed_ICU)
            patients_metadata = pd.concat([patients_metadata,metadata],axis=0)
            # Update process bar
            pbar.update(1)
        # Additional outcomes
        patients_metadata = extract_outcome(patients_metadata,los_range,readmission_range)
        patients_metadata = patients_metadata.reset_index(drop=True)
    return processed_ICUs, patients_metadata         

