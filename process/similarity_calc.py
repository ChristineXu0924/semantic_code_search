import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sentence_transformers import util

train_py = pd.read_csv('data/train_py.csv')
sample_doc = train_py['func_name'] + " " + train_py['func_documentation_string']
sample_code = list(train_py['whole_func_string'])
sample_url = train_py['func_code_url']
sample_document = list(train_py['func_documentation_string'])

def weighted_doc(query, doc_emb, model_doc, k, weight, param=3):
    que_doc_emb = model_doc.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(que_doc_emb, doc_emb)[0]
    if param*k > len(hits):
        top_hits = hits
        num_out = len(hits)
    else:
        top_hits = hits[:param*k]
        num_out = param*k

    rank = 1
    que_doc_df = pd.DataFrame()
    que_doc_score = []
    que_doc_url = []
    que_doc_rank = []
    for top_hit in top_hits:
        que_doc_score.append(weight*float(top_hit['score']))
        que_doc_url.append(train_py.iloc[top_hit['corpus_id']]['func_code_url'])
        que_doc_rank.append(rank)
        rank += 1
    que_doc_df['score'] = que_doc_score
    que_doc_df['url'] = que_doc_url
    return que_doc_df


def weighted_code(query, emb_mat, tokenizer_code, model_code, k, weight, device, param=3):
    """ 
    Inputs: natural language query, the stored embedded matrix for training code.
    Outputs: a dataframe with the top-k matches from the training code 
    
    """
    # Tokenize and encode the query
    encoded_query = tokenizer_code(query, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        query_vec = model_code(**encoded_query)[1] # Get the pooled output
    
    code_vecs = emb_mat
    
    # Calculate the cosine similarities
    query_vec - query_vec.view(1,-1).expand_as(code_vecs)
    scores = nn.functional.cosine_similarity(query_vec, code_vecs, dim=1)
    
    # Get the top 5 scores and their indices
    top_scores, top_indices = torch.topk(scores, param*k, largest=True)
    
    # Retrieve the top 5 most relevant code snippets using the indices
    top_code_snippets = [sample_url[index] for index in top_indices.cpu().numpy()]
    
    rank = 1
    que_code_df = pd.DataFrame()
    que_code_score = []
    que_code_url = []
    que_code_rank = []
    
    for score,url in zip(top_scores, top_code_snippets):
        que_code_score.append(weight*float(score.cpu()))
        que_code_url.append(url)
        que_code_rank.append(rank)
        rank += 1
    
    que_code_df['score'] = que_code_score
    que_code_df['url'] = que_code_url
    
    return que_code_df


def nlangCode(query, doc_emb, code_vecs, 
              model_doc, tokenizer_code, model_code, device, 
              weight=0.01722, k=10):
    w1 = weight
    w2 = 1-w1
    df1 = weighted_doc(query, doc_emb, model_doc, k, w1)
    df2 = weighted_code(query, code_vecs, tokenizer_code, model_code, k, w2, device)
     
    # join df1 and df2.
    combined = df1.merge(df2, how='outer', left_on='url', right_on='url').fillna(0)
    combined['avg_score'] = combined['score_x'] + combined['score_y']
    combined.sort_values(by='avg_score', inplace=True, ascending = False)
    
    combined['query']=[query for _ in range(len(combined))]
    combined['language'] = ['Python' for _ in range(len(combined))]
    return combined[:k][['language', 'query', 'url']]


    