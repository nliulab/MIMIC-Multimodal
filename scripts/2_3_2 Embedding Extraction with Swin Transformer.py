import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import SwinForImageClassification, AutoImageProcessor
from PIL import Image
from embedding_utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

# args
parser = argparse.ArgumentParser()

## read and load files
parser.add_argument('--input-icu-path', type=str)
parser.add_argument('--input-metadata-path', type=str)
parser.add_argument('--output-path', type=str)

## parameters
parser.add_argument('--model-name', type=str, default="microsoft/swin-base-patch4-window7-224-in22k")
parser.add_argument('--local-dir', type=str, default='D:/')


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
    
    # Get indexes of ICU stays with Chest X-ray data
    idx_list = []
    for i in range(len(icus)):
        icu = icus[i]
        img_metadata = icu.images['metadata']
        # Append index to the list if metadata table for images is not empty
        if not img_metadata.empty:
            idx_list.append(i)
            
    # Load the Swin Transformer-Base model
    model_name = args.model_name
    model = SwinForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    auto_processor = AutoImageProcessor.from_pretrained(model_name)
    # Create a feature extractor
    feature_extractor = SwinFeatureExtractor(model)
    
    print('Embedding Extraction')
    local_dir = args.local_dir
    with tqdm(total=len(idx_list)) as pbar:
        all_embeddings = []
        for idx in idx_list:
            icu = icus[idx]
            img_metadata = icu.images['metadata']
            df_img = img_metadata.merge(icu.images['image_path'],how='left',on=['dicom_id','study_id','subject_id'])
            df_img['local_jpg_file'] = df_img['path'].apply(lambda x: os.path.join(local_dir, os.path.splitext(x)[0] + '.jpg'))
            # print(idx)
            # display(df_img)
            embeddings = []
            for _,row in df_img.iterrows():
                path = row['local_jpg_file']
                image = Image.open(path).convert("L")  # Convert to grayscale
                image = Image.merge("RGB", (image, image, image))  # Convert to 3-channel RGB
                # processed_image = preprocess(image).unsqueeze(0)  # Add batch dimension
                # print(image.shape)
                processed_image = auto_processor(images=image, return_tensors="pt")
                inputs = processed_image['pixel_values']
                
                # Extract embeddings
                with torch.no_grad():
                    image_embedding = feature_extractor(inputs)
                # Convert to numpy array and append to list
                embeddings.append(image_embedding.squeeze(0).numpy())
            # Compute the mean of the embeddings
            averaged_embedding = np.mean(embeddings, axis=0)
            # Append to list of embeddings from all ICU stays with Chest X-ray
            all_embeddings.append(averaged_embedding)
            # Update progress
            pbar.update(1)

    # Convert the list of arrays into a DataFrame
    swin_transformer_embeddings= pd.DataFrame(all_embeddings)
    swin_transformer_embeddings.index = idx_list
    swin_transformer_embeddings.index.name = 'Index'
    # Export
    save_npz(swin_transformer_embeddings,os.path.join(output_path,f'swin_transformer_embeddings.npz'))