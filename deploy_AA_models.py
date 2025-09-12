import os
import json
import pandas as pd

import torch
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from sklearn.metrics import classification_report
from transformers import Trainer, AutoModelForSequenceClassification


parser = argparse.ArgumentParser(description="Deploy an AA model.")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for training")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def get_text_encodings(model_name, texts, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return  tokenizer(texts, truncation=True, 
                      padding="max_length", 
                      max_length=max_length)


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataset(model_name, texts,
                max_length, labels):
    
    encodings = get_text_encodings(model_name, texts, 
                                   max_length)

    dataset = CustomDataset(encodings, labels)
    return dataset


def get_model_and_trainer(ckpt_dir):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    trainer = Trainer(model=model)
    return trainer


def print_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred, zero_division=0))


def deploy_an_AA_model(ckpt_dir, deploy_fp, 
                       text_col="writing", 
                       top_k=10, overwrite=False):
    
    ckpt_dir_parent = os.path.dirname(ckpt_dir)

    with open(os.path.join(ckpt_dir_parent, "args.json"), "r") as f:
        args = json.load(f)
    
    model_name = args["model_name"]
    max_length = args["max_length"]
    model_name__ = model_name.split('/')[-1]

    df = pd.read_csv(deploy_fp)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in the DataFrame.")

    df[text_col] = df[text_col].fillna("SOMETHING_WRONG")
    
    if f"{model_name__}-AA-top_k-predictions" in df.columns and not overwrite:
        print(f"Column '{model_name__}-AA-top_k-predictions' already exists in the DataFrame. "
              f"Set 'overwrite=True' to overwrite it.")
        return

    labels = [0] * len(df)  # Dummy labels, not used in prediction
    dataset = get_dataset(model_name, 
                          df[text_col].tolist(), 
                          max_length, labels)
    
    trainer = get_model_and_trainer(ckpt_dir)
    predictions = trainer.predict(dataset)
    logits = predictions.predictions  # This contains the raw logits output
    # Convert logits to probabilities using softmax
    probabilities = softmax(torch.tensor(logits), dim=1)

    topk_values, topk_indices = torch.topk(probabilities, k=top_k, dim=1)

    # Convert to Python lists for further use
    top_k_probs = topk_values.tolist()
    top_k_preds = topk_indices.tolist()

    df = pd.read_csv(deploy_fp) # reload to avoid overwriting
    df[text_col] = df[text_col].fillna("SOMETHING_WRONG")
    df[f"{model_name__}-AA-top_k-probabilities"] = top_k_probs
    df[f"{model_name__}-AA-top_k-predictions"] = top_k_preds
    df.to_csv(deploy_fp, index=False)
    print(f"Deployment completed. Results saved to {deploy_fp}")


def main():
    models = ["AA_models/longformer-base-4096", 
              "AA_models/ModernBERT-base"]
    datasets = ["CCAT50", "enron", "reddit", "blog"]

    for model in models:
        for dataset in datasets:
            dir_path = os.path.join(model, dataset)

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
                    deploy_an_AA_model(ckpt_dir, llm_fp,
                                    text_col="writing", 
                                    top_k=10, overwrite=args.overwrite)

if __name__ == "__main__":
    main()