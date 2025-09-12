import math
from litellm import completion
from nltk.tokenize import word_tokenize
import uuid
import json
import random
from time import sleep
import os
from os import listdir, walk
from os.path import isfile, join
import pandas as pd


def get_completion(prompt, 
                   model="openai/gpt-4.1-mini-2025-04-14", 
                   temperature=0, max_tries=5):
    
    for _ in range(max_tries):
        try:
            # Call the completion function with the provided parameters
            response = completion(
                model=model, temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        
        except Exception as e:
            print(f"Error: {e}")
            sleep(10)
            continue
    
    return "SOMETHING_WRONG"


def count_words(text):
    return len(word_tokenize(text))


def round_up_to_nearest_10(n):
    return math.ceil(n / 10) * 10


def list_writing_samples(samples):
    return '\n\n'.join([f'Sample#{ix+1}\n\n{sample.strip()}' 
                      for ix, sample in enumerate(samples)])


def save_dict_to_json(dict_obj, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_obj, f)


def load_dict_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def generate_random_id():
    return str(uuid.uuid4())



def get_filepathes_from_dir(file_dir, include_sub_dir=False,
                            file_format=None, shuffle=False):
    
    if include_sub_dir:
        filepathes = []
        for root, _, files in walk(file_dir, topdown=False):
            for f in files:
                filepathes.append(join(root, f))
    else:
        filepathes = [join(file_dir, f) for f in listdir(file_dir)
                      if isfile(join(file_dir, f))]
        
    if file_format:
        if not isinstance(file_format, (str, list, tuple)):
            raise TypeError("file_format must be str, list or tuple.")
        file_format = tuple(file_format) if isinstance(file_format, list) else file_format
        format_checker = lambda f: f.endswith(file_format)
        filepathes = list(filter(format_checker, filepathes))

    if shuffle:
        random.shuffle(filepathes)
        
    return filepathes


def find_directories_with_file(root_dir, target_filename):
    matching_dirs = []

    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        if target_filename in filenames:
            matching_dirs.append(dirpath)

    return matching_dirs


def align_df1_to_df2(df1, df2, col1, col2, col3):
    """
    Aligns df1 to df2 based on matching values in columns `col1,` `col2` and `col3`,
    handling duplicates in both dataframes.

    Returns a new DataFrame with the same number of rows and order as df2,
    by selecting matching rows from df1.
    """
    df1_temp = df1.copy()
    df1_temp['_used'] = False
    matched_rows = []

    for _, row in df2.iterrows():
        match = df1_temp[
            (df1_temp[col1] == row[col1]) &
            (df1_temp[col2] == row[col2]) &
            (df1_temp[col3] == row[col3]) &
            (~df1_temp['_used'])
        ]

        if match.empty:
            return None

        first_match_idx = match.index[0]
        matched_rows.append(df1_temp.loc[first_match_idx])
        df1_temp.at[first_match_idx, '_used'] = True

    result = pd.DataFrame(matched_rows).drop(columns=['_used']).reset_index(drop=True)
    return result