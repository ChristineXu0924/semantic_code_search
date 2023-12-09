import re
import os
from data import sample_code


output_dir = 'cache/cleaned_code_string'
os.makedirs(output_dir, exist_ok=True)

def clean_code(code_str):
    # for individual code string
    pattern_doc = r'\"\"\"(.*?)\"\"\"'
    pattern_comm = r'#[^\n]*'
    clean_code = re.sub(pattern_doc, '', code_str, flags=re.DOTALL)
    clean_code = re.sub(pattern_comm, '', clean_code, flags=re.DOTALL)

    return clean_code

for i, code in enumerate(sample_code):
    cleaned_code = clean_code(code)
    file_path = os.path.join(output_dir, f'code_{i}.py')
    with open(file_path, 'w') as file:
        file.write(cleaned_code)

print(f"Files are saved in {output_dir}")