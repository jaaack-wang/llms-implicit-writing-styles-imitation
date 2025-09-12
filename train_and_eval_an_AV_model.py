import os
import torch
import json
import argparse
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import classification_report
from torch.nn.functional import softmax


def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an AV model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
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
    parser.add_argument("--resume_from_checkpoint", type=str, default="True", help="Resume training from checkpoint")
    parser.add_argument("--do_toy_run", action="store_true", help="Run a toy example for debugging")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for training")

    return parser.parse_args()


def load_AV_dataset(data_dire):
    train_df = pd.read_csv(os.path.join(data_dire, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_dire, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_dire, "test.csv"))

    return train_df, valid_df, test_df


def bool_str_to_bool(value):
    return value.lower() in ('true', '1', 'yes', 'y', 't')


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


def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_metric.add_batch(predictions=predictions, references=labels)
    return f1_metric.compute()


def main():
    args = get_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Load the model for sequence classification
    if args.data_dir.endswith("/"):
        args.data_dir = args.data_dir[:-1]

    dataset = args.data_dir.split("/")[-1]
    model_output_dir = f"./AV_models/{args.model_name.split('/')[-1]}/" + dataset
    os.makedirs(model_output_dir, exist_ok=True)

    # save the args in a json file
    args_dict = vars(args)
    with open(os.path.join(model_output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
        print(f"Saved args to {os.path.join(model_output_dir, 'args.json')}")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.cuda.empty_cache()

    # Load the datasets
    train_df, valid_df, test_df = load_AV_dataset(args.data_dir)
    print(f"Loaded datasets from {args.data_dir}")

    # If do_toy_run is set, sample a small fraction of the dataset
    if args.do_toy_run:
        train_df = train_df.sample(frac=0.1).reset_index(drop=True)
        valid_df = valid_df.sample(frac=0.1).reset_index(drop=True)
        print("Running a toy example (10% of original train/valid samples) for debugging")

    # Load the tokenizer and tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Loaded tokenizer from {args.model_name}")

    train_encodings = tokenizer(list(train_df['text1']), list(train_df['text2']), 
                                truncation=True, padding="max_length", max_length=args.max_length)
    valid_encodings = tokenizer(list(valid_df['text1']), list(valid_df['text2']), truncation=True, 
                                padding="max_length", max_length=args.max_length)
    test_encodings = tokenizer(list(test_df['text1']), list(test_df['text2']), truncation=True, 
                               padding="max_length", max_length=args.max_length)

    # Create datasets
    train_dataset = CustomDataset(train_encodings, train_df['label'].tolist())
    valid_dataset = CustomDataset(valid_encodings, valid_df['label'].tolist())
    test_dataset = CustomDataset(test_encodings, test_df['label'].tolist())


    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, 
                                                               num_labels=2, 
                                                               device_map="auto")

    training_args = TrainingArguments(
        output_dir=model_output_dir,  # Output directory
        fp16=bool_str_to_bool(args.fp16),  # Use mixed precision training
        num_train_epochs=args.num_train_epochs,  # Total number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # Batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps=args.warmup_steps,  # Number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # Strength of weight decay
        learning_rate=args.learning_rate,  # Initial learning rate
        save_total_limit=args.save_total_limit,  # Limit the total amount of checkpoints
        logging_steps=args.logging_steps,  # Log every X updates steps
        eval_strategy=args.evaluation_strategy,  # Evaluation strategy to adopt during training
        save_strategy=args.evaluation_strategy,  # Save strategy to adopt during training
        load_best_model_at_end= bool_str_to_bool(args.load_best_model_at_end),  # Load the best model at the end of training
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Create the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],  # Early stopping callback
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

    y_test = test_df.label.tolist()
    print(classification_report(y_test, y_pred))

    model_name = args.model_name.split('/')[-1]
    # with open(os.path.join(args.data_dir, f"{model_name}-classification_report.txt"), "w") as f:
    #     f.write(classification_report(y_test, y_pred))

    logits = predictions.predictions  # This contains the raw logits output
    # Convert logits to probabilities using softmax
    probabilities = softmax(torch.tensor(logits), dim=1).tolist()
    test_df[f"{model_name}-AV-prediction"]=y_pred
    test_df[f"{model_name}-AV-probabilities"] = [prob[1] for prob in probabilities]
    test_df.to_csv(os.path.join(args.data_dir, "test.csv"), index=False)
    print(f"Test predictions saved to {os.path.join(args.data_dir, 'test.csv')}")


if __name__ == "__main__":
    main()
