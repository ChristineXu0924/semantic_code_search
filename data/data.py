from datasets import load_dataset
import pandas as pd
import numpy as np

path = "data/train_py.csv"
train_data = pd.read_csv(path)

sample_doc = train_data['func_name'] + " " + train_data['func_documentation_string']
sample_code = list(train_data['whole_func_string'])
sample_url = train_data['func_code_url']
sample_document = list(train_data['func_documentation_string'])

