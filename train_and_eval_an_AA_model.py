import os
import json
import argparse
import pandas as pd

import torch
from torch.utils.data import Dataset

import numpy as np
import evaluate  

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback


from sklearn.metrics import classification_report
from torch.nn.functional import softmax



def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an AA model.")
    parser.add_argument("--training_df_fp", type=str, required=True, help="Filepath for the training dataset")
    parser.add_argument("--test_df_fp", type=str, required=True, help="Filepath for the test dataset")

    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096", help="Name of the model to be used")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of the input sequences")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--load_best_model_at_end", type=str, default="True", help="Load the best model at the end of training")
    parser.add_argument("--fp16", type=str, default="True", help="Use mixed precision training")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Limit the total amount of checkpoints")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--do_toy_run", action="store_true", help="Run a toy example for debugging")
    parser.add_argument("--resume_from_checkpoint", type=str, default="True", help="Resume training from checkpoint")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for training")

    return parser.parse_args()


def get_author_map(df, author_col="author"):
    author_map = {author: i for i, author in 
    enumerate(df[author_col].unique())}
    return author_map


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


def bool_str_to_bool(value):
    return value.lower() in ('true', '1', 'yes', 'y', 't')


def compute_metrics(eval_pred):
    f1_score = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_score.add_batch(predictions=predictions, references=labels)
    return f1_score.compute(average="weighted")


def main():
    args = get_args()
    # Check if the dataset names match
    # between training and evaluation datasets
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    dataset = args.training_df_fp.split("/")[-1].split(".")[0].split("_")[0]
    dataset_ = args.test_df_fp.split("/")[-1].split(".")[0].split("_")[0]
    assert dataset == dataset_, f"Dataset name mismatch: {dataset} != {dataset_}"

    # save the args in a json file
    model_output_dir = f"./AA_models/{args.model_name.split('/')[-1]}/" + dataset
    os.makedirs(model_output_dir, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(model_output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
        print(f"Saved args to {os.path.join(model_output_dir, 'args.json')}")

    # Load the training and test datasets
    df = pd.read_csv(args.training_df_fp)
    test_df = pd.read_csv(args.test_df_fp)

    # Get the author map
    if "AA-label" not in df.columns:
        author_map = get_author_map(df)
        df["AA-label"] = df["author"].map(author_map)
        test_df["AA-label"] = test_df["author"].map(author_map)
        df.to_csv(args.training_df_fp, index=False)
        test_df.to_csv(args.test_df_fp, index=False)
        print(f"Appended author labels to {args.training_df_fp} and {args.test_df_fp}") 
    
    if args.do_toy_run:
        df = df.sample(1000, random_state=42).reset_index(drop=True)
        print(f"Running a toy example with {len(df)} samples")

    # Split the training data into train and validation sets
    train_df, valid_df = train_test_split(df, test_size=0.2, 
                                          random_state=42, 
                                          stratify=df["AA-label"])
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Load the tokenizer and tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_encodings = tokenizer(list(train_df['text']), 
                                truncation=True, padding="max_length",
                                max_length=args.max_length)
    valid_encodings = tokenizer(list(valid_df['text']), 
                                truncation=True, padding="max_length",
                                max_length=args.max_length)
    test_encodings = tokenizer(list(test_df['text']),
                               truncation=True, padding="max_length",
                               max_length=args.max_length)
    train_dataset = CustomDataset(train_encodings, train_df['AA-label'])
    valid_dataset = CustomDataset(valid_encodings, valid_df['AA-label'])
    test_dataset = CustomDataset(test_encodings, test_df['AA-label'])

    # Load the model and train it
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, device_map="auto",
                                                               num_labels=len(df["AA-label"].unique()))
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,  # output directory
        fp16=bool_str_to_bool(args.fp16),  # Use mixed precision training
        num_train_epochs=args.num_train_epochs,  # total # of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        # number of updates steps to accumulate before performing a backward/update pass
        gradient_accumulation_steps=args.gradient_accumulation_steps,  
        warmup_steps=args.warmup_steps,  # Number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # Strength of weight decay
        learning_rate=args.learning_rate,  # Initial learning rate
        save_total_limit=args.save_total_limit,  # Limit the total amount of checkpoints
        logging_steps=args.logging_steps,  # Log every X updates steps
        evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
        save_strategy=args.evaluation_strategy,  # save strategy to adopt during training
        load_best_model_at_end=args.load_best_model_at_end,  # load the best model when finished training
        metric_for_best_model="eval_loss",  # use f1 score to compare models
        greater_is_better=False,  # f1 score should be greater
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],  # early stopping callback
    )

    # Train the model
    resume_from_checkpoint = bool_str_to_bool(args.resume_from_checkpoint)
    if any(["checkpoint" in file for file in os.listdir(model_output_dir)]) and resume_from_checkpoint:
        print("Resuming from the latest checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train(resume_from_checkpoint=False)

    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)

    y_test = test_df["AA-label"].tolist()
    print(classification_report(y_test, y_pred))

    logits = predictions.predictions  # This contains the raw logits output
    # Convert logits to probabilities using softmax
    probabilities = softmax(torch.tensor(logits), dim=1).tolist()

    model_name = args.model_name.split('/')[-1]
    test_df[f"{model_name}-AA-prediction"]=y_pred
    test_df[f"{model_name}-AA-probabilities"] = [prob[1] for prob in probabilities]
    test_df.to_csv(args.test_df_fp, index=False)
    print(f"Predictions and probabilities by {args.model_name} saved to {args.test_df_fp}")


if __name__ == "__main__":
    main()