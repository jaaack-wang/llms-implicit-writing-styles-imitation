import os
import json
import pandas as pd

import torch
import argparse
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import Trainer, AutoModelForSequenceClassification


parser = argparse.ArgumentParser(description="Deploy an AA model.")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for training")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def get_text_encodings(model_name, texts1, texts2, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(texts1, texts2, 
                     truncation=True, 
                     padding="max_length", 
                     max_length=max_length)


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataset(model_name, texts1, texts2,
                max_length, labels):
    
    encodings = get_text_encodings(model_name, 
                                   texts1, texts2, 
                                   max_length)

    dataset = CustomDataset(encodings, labels)
    return dataset


def get_model_and_trainer(model_load_file):
    model = AutoModelForSequenceClassification.from_pretrained(model_load_file)
    trainer = Trainer(model=model)
    return trainer


def deploy_an_AV_model(ckpt_dir, 
                       deploy_fp1, 
                       deploy_fp2,
                       text_col1,
                       text_col2,
                       overwrite=False):
    ckpt_dir_parent = os.path.dirname(ckpt_dir)

    with open(os.path.join(ckpt_dir_parent, "args.json"), "r") as f:
        args = json.load(f)
    
    model_name = args["model_name"]
    max_length = args["max_length"]
    
    df1 = pd.read_csv(deploy_fp1)
    df2 = pd.read_csv(deploy_fp2)

    assert df1.shape[0] == df2.shape[0], \
        f"DataFrames must have the same number of rows. " \
        f"Got {df1.shape[0]} and {df2.shape[0]} rows."

    model_name__ = model_name.split('/')[-1]
    if f"{model_name__}-AV-prediction" in df2.columns and not overwrite:
        print(f"Column '{model_name__}-prediction' already exists in the DataFrame. "
              f"Set 'overwrite=True' to overwrite it.")
        return

    if text_col1 not in df1.columns:
        raise ValueError(f"Column '{text_col1}' not found in the DataFrame.")
    if text_col2 not in df2.columns:
        raise ValueError(f"Column '{text_col2}' not found in the DataFrame.")
    
    df1[text_col1] = df1[text_col1].fillna("SOMETHING_WRONG")
    df2[text_col2] = df2[text_col2].fillna("SOMETHING_WRONG")

    labels = [0] * len(df1)  # Dummy labels, not used in prediction
    dataset = get_dataset(model_name, df1[text_col1].tolist(), 
                          df2[text_col2].tolist(), max_length, labels)
    
    trainer = get_model_and_trainer(ckpt_dir)
    predictions = trainer.predict(dataset)
    y_pred = predictions.predictions.argmax(-1)
    logits = predictions.predictions  # This contains the raw logits output
    # Convert logits to probabilities using softmax
    probabilities = softmax(torch.tensor(logits), dim=1).tolist()

    df2 = pd.read_csv(deploy_fp2) # reload to avoid overwriting
    df2[text_col2] = df2[text_col2].fillna("SOMETHING_WRONG")
    df2[f"{model_name__}-AV-prediction"] = y_pred
    df2[f"{model_name__}-AV-probabilities"] = probabilities
    df2.to_csv(deploy_fp2, index=False)
    print(f"Deployment completed. Results saved to {deploy_fp2}")


def main():
    models = ["AV_models/longformer-base-4096", 
              "AV_models/ModernBERT-base"]
    datasets = ["CCAT50", "enron", "reddit", "blog"]

    for model in models:
        for dataset in datasets:
            dir_path = os.path.join(model, dataset + "_AV_datasets")
            
            if not os.path.exists(dir_path):
                print(f"Directory {dir_path} does not exist.")
                continue

            ckpt_dir_names = [dn for dn in os.listdir(dir_path) if dn.startswith("checkpoint-")]
            
            if not ckpt_dir_names:
                print(f"No checkpoints found in {dir_path}.")
                continue
            
            # select the latest checkpoint
            ckpt_dir_names.sort(key=lambda x: int(x.split("-")[1]))
            ckpt_dir = os.path.join(dir_path, ckpt_dir_names[-1])

            for setting in [1, 2, 3, 4, 5, 6]:
                dataset_dir = os.path.join("LLM_writing", f"Setting{setting}", dataset)
                if not os.path.exists(dataset_dir):
                    print(f"Directory for Setting {setting} and dataset {dataset} does not exist.")
                    continue
                    
                prompt_fp = os.path.join(dataset_dir, "prompts.csv")
                if not os.path.exists(prompt_fp):
                    print(f"Prompts file not found in {dataset_dir}.")
                    continue
                df_prompts = pd.read_csv(prompt_fp)
                
                llm_fps = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) 
                        if f.endswith(".csv") and f != "prompts.csv"]
                
                for llm_fp in llm_fps:
                    llm_df = pd.read_csv(llm_fp)

                    if len(df_prompts) != len(llm_df):
                        print(f"Length mismatch between prompts and LLM-generated writing for {llm_fp}.")
                        continue
                    
                    print(f"===> Deploying model {model} on {llm_fp}")
                    deploy_an_AV_model(ckpt_dir, prompt_fp, 
                                       llm_fp, text_col1="text",
                                       text_col2="writing",
                                       overwrite=args.overwrite)


if __name__ == "__main__":
    main()
