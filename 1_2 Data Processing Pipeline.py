import numpy as np
import pandas as pd
import pickle
from data_utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

# args
parser = argparse.ArgumentParser()

## read and load files
parser.add_argument('--input-path', type=str)
parser.add_argument('--number', type=int, default=None,
                    help='Number of patient ICU stays to be processed; if None, process all')
parser.add_argument('--output-pkl-path', type=str, help='Output path for processed ICU stay data')
parser.add_argument('--output-csv-path', type=str, help='Output path for metadata table')

## inclusion criteria
parser.add_argument('--age-lower', type=int, default=0, help='Lower bound of patient age')
parser.add_argument('--age-upper', type=int, default=150, help='Upper bound of patient age')
parser.add_argument('--drop-missing-modalities', nargs='+',
                    default=[False, False, False],
                    help='Drop objects with modalities missing (a list as input for time-series,image and text)')
parser.add_argument('--los-lower', type=int, default=0, help='Lower bound of patient length of stay')

## data pre-processing
parser.add_argument('--start-diff', type=int, default=None,
                    help='Time difference between the start of information collection period and ICU admission time(positive/negative)')
parser.add_argument('--end-diff', type=int, default=None,
                    help='Time difference between the end of information collection period and ICU discharge time(positive only)')
parser.add_argument('--tabular-variables', nargs='+', 
                    default=['subject_id','hadm_id','stay_id','age','gender','race','marital_status','language',
                             'insurance'], help='List of tabular variables included')
parser.add_argument('--vital-signs-variables', nargs='+', 
                    default=['Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry', 
                             'Non Invasive Blood Pressure systolic', 'Non Invasive Blood Pressure diastolic', 
                             'Non Invasive Blood Pressure mean', 'Arterial Blood Pressure systolic',
                             'Arterial Blood Pressure diastolic','Arterial Blood Pressure mean',
                             'GCS - Eye Opening', 'GCS - Verbal Response', 'GCS - Motor Response',
                             'Temperature Fahrenheit'], help='List of vital signs variables included')
# parser.add_argument('--lab-variables', nargs='+', 
#                     default=['Glucose','Potassium','Sodium','Chloride','Hematocrit','Creatinine','Urea Nitrogen',
#                              'Hemoglobin','Bicarbonate','Anion Gap','Platelet Count','White Blood Cells','MCHC',
#                              'Red Blood Cells','MCV','MCH','RDW','Magnesium','Phosphate','Calcium, Total'],
#                     help='List of lab test variables included')
parser.add_argument('--sorted', action='store_false')
parser.add_argument('--ascending', action='store_false')
parser.add_argument('--los-range', type=int, default=3, help='Convert length of stay to binary outcome by a customized time range(days)')
parser.add_argument('--readmission-range', type=int, default=30, help='Time range for patient re-admission in days')

args = parser.parse_args()

if __name__ == "__main__":
    
    print(args)
    input_path = args.input_path
    file_list = os.listdir(input_path)
    
    print('Read patient ICU information from pickle files')
    if args.number:
        patient_icus = load_patient_ICU_objects(input_path,file_list[0:args.number])
    else:
        patient_icus = load_patient_ICU_objects(input_path,file_list)
        
    print('Process patient ICU information')
    processed_icus, metadata = process_all(patient_icus,args.age_lower,args.age_upper,args.drop_missing_modalities,args.los_lower,
                                                        args.start_diff,args.end_diff,args.tabular_variables,
                                                        args.vital_signs_variables, args.sorted,args.ascending,
                                                        args.los_range, args.readmission_range)
    # # Example
    # sample = processed_icus[0]
    # for attribute, value in sample.__dict__.items():
    #     print(attribute)
    #     print(value)
    output_pkl_path = args.output_pkl_path
    pickle.dump(processed_icus,open(output_pkl_path,'wb'))
    output_csv_path = args.output_csv_path
    metadata.to_csv(output_csv_path)
    print('Completed')