import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from evaluation import relevanceeval
from process import similarity_calc

SEED = 42
TORCHSEED = 1


def set_seed(seed, torch_seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.manual_seed(seed)

    
def evaluate(doc_emb, code_emb, model_doc, tokenizer_code, model_code, device):
    out_dfs = []
    queries = pd.read_csv('evaluation/queries.csv')
    for query in queries['query'].values:
        out_df = similarity_calc.nlangCode(query, doc_emb, code_emb, model_doc, tokenizer_code, model_code, device)
        out_dfs.append(out_df)
    trainset_result = pd.concat(out_dfs, ignore_index=True)
    trainset_result.to_csv('evaluation/train_py_result.csv')
        
    return None


def main():
    
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
         
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #set seed
    set_seed(SEED, TORCHSEED)
    
    #load in embeddings
    doc_emb = torch.load('cache/doc_emb.pt', map_location=torch.device(device))
    code_emb = torch.load('cache/clean_code_emb.pt',map_location=torch.device(device))
    
    #load in models
    model_doc = SentenceTransformer('all-MiniL M-L6-v2')
    tokenizer_code = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model_code = RobertaModel.from_pretrained("model/python_model").to(device)
    
    #produce output
    print('Start calculating the final results. The process might take around 3 minutes')
    evaluate(doc_emb, code_emb, model_doc, tokenizer_code, model_code, device)
    
    
    
    
if __name__ == "__main__":
    main()
    