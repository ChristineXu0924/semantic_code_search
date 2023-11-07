# Q1 Final Project

## Architecture
On a high level, our search will first embed all documents (i.e. Python code, documentation) into a vector space. We plan to use the Sentence Transformer 
“all-MiniLM-L6-v2” on both documentation and code functions. Once a natural language query is inputted, the query will also be embedded in the same vector 
space as all the documents. For every document vector, the similarities for documentation and code will be calculated and combined with the query vector and the 
top 5 documents with the closest similarities will be returned. In terms of weighting, depending on which feature does better (our current hypothesis is that 
documentation will perform better), we will hardcode a higher weight to that feature. 

## Algorithms
For the embeddings and models, we grabbed off-the-shelf search tools like the sentence-transformer (“all-MiniLM-L6-v2”). Both of these models embed (query, code) 
and (query, documentation) pairs into vectors with the same dimension. Then the similarity scores will be calculated using cosine-similarity (for sentence-transformer). 
The result will be sorted in descending order, with the largest value indicating the most relevant result. 

## Dataset
We are only using our search on the available Python data within the CodeSearchNet Challenge dataset. The reason for this is to be able to focus on deliberately 
choosing more optimal models for different features as well as for fine tuning reasons.

## Evaluation
We will be using NDCG to evaluate our search results. Currently the benchmark we used is the annotated query-code from CodeSeachNet, and the score calculated through 
checking the position of the proved-relevant code in our results. 

## Improvements
- Apply different models to different features – Sentence transformer (“all-MiniLM-L6-v2”) on documentation, CodeBERT (“microsoft/codebert-base”) on code functions
  - Cosine similarity to calculate similarity for the sentence transformer and Einstein summation to calculate similarity for CodeBERT
- Try testing the whole training set on distributed computing platforms
- Extend search to all 6 programming languages, be able to train separate features and model selection for each language
