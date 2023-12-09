# Q1 Final Project

## Result Reproduction
### Preprocessed Data and Embeddings
Our original dataset and trained embeddings are stored in this [google drive folder](https://drive.google.com/drive/folders/1I0XicfB_W7FijCDsjLv6CCxep_y8fqgQ?usp=drive_link) due to the scale of these data. The code with detailed process can be found in the notebook **linear_semantic-codeBERT.ipynb**.

Please download the files and put into corresponding folders (data and cache) in order for the command line argument to successfully process.
### Dependency and Enviornment Setup
After cloning the github repo to local repository and installing anaconda, cd into the directory and initiat a conda enviornemtn with command
```
conda create -n enviornment_name
conda activate enviornment_name
conda env create -f enviornment.yml
```

### Reproduce 
In terminal, cd into the corresponding directory and run the following command to get the evaluation result. 
```
python run.py
cd evaluation
python relevanceeval.py annotationStore.csv train_py_result.csv
```

## Architecture
On a high level, our search will first embed all documents (i.e. Python code, documentation) into a vector space. For our baseline model, we use the Sentence Transformer 
“all-MiniLM-L6-v2” on both documentation and code functions. We also use create a second model that uses the same Sentence Transformer on documentation and CodeBERT on code and a third model that uses BM25 on documentation and CodeBERT on code. Once a natural language query is inputted, the query will also be embedded in the same vector 
space as all the documents. For every document vector, the similarities for documentation and code will be calculated and combined with the query vector and the 
top _k_ documents with the closest similarities will be returned. In terms of weighting, depending on which feature does better (our current hypothesis is that 
documentation will perform better), we will hardcode a higher weight to that feature. 

## Algorithms
For the embeddings and models, we grab off-the-shelf search tools like the sentence-transformer (“all-MiniLM-L6-v2”), CodeBERT, and BM25. These models embed (query, code) 
and (query, documentation) pairs into vectors with the same dimension. Then the similarity scores will be calculated using either cosine-similarity or the model's own similarity scoring. 
The result will be sorted in descending order, with the largest value indicating the most relevant result. 

## Dataset
We are only using our search on the available Python data within the CodeSearchNet Challenge dataset. The reason for this is to be able to focus on deliberately 
choosing more optimal models for different features as well as for fine tuning reasons.

## Evaluation
We will be using NDCG to evaluate our search results. Currently we use the NDCG scores from the original CodeSearchNet paper as a reference and use our baseline model's NDCG score as a benchmark.

## Improvements/Reach Goals
- Apply different models to different features – Sentence transformer (“all-MiniLM-L6-v2”) on documentation, CodeBERT (“microsoft/codebert-base”) on code functions
  - Cosine similarity to calculate similarity for the sentence transformer and Einstein summation to calculate similarity for CodeBERT
- Try testing the whole training set on distributed computing platforms
- Extend search to all 6 programming languages, be able to train separate features and model selection for each language
