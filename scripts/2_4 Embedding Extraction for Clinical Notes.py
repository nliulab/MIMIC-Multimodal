import os
import numpy as np
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
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

parser.add_argument('--emb-technique', type=str, help='openai/radbert')

## Parameters
parser.add_argument('--api-key', type=str)
parser.add_argument('--model-name', type=str, help='StanfordAIMI/RadBERT or text-embedding-3-large')

args = parser.parse_args()

def extract_openai_text_embedding(text, model_name, dimension=1024):
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model_name, dimensions=dimension).data[0].embedding

if __name__ == "__main__":
    
    # --input-icu-path './Data/processed_icu_24h.pkl' --input-metadata-path './Data/metadata_24h.csv' --output-path 'Embeddings/'
    print(args)
    input_icu_path = args.input_icu_path
    input_metadata_path = args.input_metadata_path
    output_path = args.output_path
    
     # Load data
    icus = pickle.load(open(input_icu_path,'rb'))
    icus_metadata = pd.read_csv(input_metadata_path,index_col=0)
    
    # Get indexes of ICU stays with radiology reports data
    idx_list = []
    for i in range(len(icus)):
        icu = icus[i]
        rr = icu.notes['radiology']
        # Append index to the list if metadata table for images is not empty
        if not rr.empty:
            idx_list.append(i)

    # Choose embedding technique
    emb_technique = args.emb_technique
    
    if emb_technique == 'openai':
        api_key = args.api_key
        model_name = args.model_name
        client = OpenAI(api_key=api_key)
        
        with tqdm(total=len(idx_list)) as pbar:
            all_embeddings = []
            for idx in idx_list:
                icu = icus[idx]
                rr = icu.notes['radiology']['text'].unique()
                annotated_text = [f"Report {i}:\n{text}" for i, text in enumerate(rr, 1)]
                combined_text = "".join(annotated_text)
                # print(idx)
                # print(combined_text)
                embedding = extract_openai_text_embedding(combined_text, model_name)
                all_embeddings.append(np.array(embedding))
                pbar.update(1)
        
        openai_embeddings= pd.DataFrame(all_embeddings)
        openai_embeddings.index = idx_list
        openai_embeddings.index.name = 'Index'
        # Export
        save_npz(openai_embeddings,os.path.join(output_path,f'openai_embeddings.npz'))
        
    elif emb_technique == 'radbert':
        model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        with tqdm(total=len(idx_list)) as pbar:
            all_embeddings = []
            for idx in idx_list:
                icu = icus[idx]
                rr = icu.notes['radiology']['text'].unique()
                combined_text = combine_text(rr)
                embedding = extract_radbert_text_embedding(combined_text,model,tokenizer)
                all_embeddings.append(np.array(embedding))
                pbar.update(1)
        
        radbert_embeddings= pd.DataFrame(np.vstack(all_embeddings))
        radbert_embeddings.index = idx_list
        radbert_embeddings.index.name = 'Index'
        # Export
        save_npz(radbert_embeddings,os.path.join(output_path,f'radbert_embeddings.npz'))