import os
import pandas as pd
from random import sample
from itertools import combinations, product
from sklearn.model_selection import train_test_split


def construct_AV_dataset(df, 
                         author_col="author", 
                         text_col="text", 
                         num_of_samples=100000, 
                         pos_ratio=0.4):
    cols = ["author1", "author2", "text1", "text2", "label"]
    pos_samples = []
    neg_samples = []

    # Group texts by author
    author_groups = df.groupby(author_col)

    # Generate all possible positive pairs (same author)
    for author, group in author_groups:
        indices = group.index.tolist()
        if len(indices) >= 2:
            pos_samples.extend(combinations(indices, 2))

    # Generate all possible negative pairs (different authors)
    authors = df[author_col].unique()
    author_indices = {author: df[df[author_col] == author].index.tolist() for author in authors}
    author_list = list(author_indices.keys())

    for i in range(len(author_list)):
        for j in range(i+1, len(author_list)):
            idx1 = author_indices[author_list[i]]
            idx2 = author_indices[author_list[j]]
            neg_samples.extend(product(idx1, idx2))

    # Shuffle and sample
    pos_needed = int(num_of_samples * pos_ratio)
    neg_needed = num_of_samples - pos_needed

    pos_pairs = sample(pos_samples, min(pos_needed, len(pos_samples)))
    neg_pairs = sample(neg_samples, min(neg_needed, len(neg_samples)))

    # Construct output
    out = []
    for ix1, ix2 in pos_pairs:
        out.append([df.at[ix1, author_col], df.at[ix2, author_col], df.at[ix1, text_col], df.at[ix2, text_col], 1])
    for ix1, ix2 in neg_pairs:
        out.append([df.at[ix1, author_col], df.at[ix2, author_col], df.at[ix1, text_col], df.at[ix2, text_col], 0])

    out_df = pd.DataFrame(out, columns=cols)
    out_df = out_df.sample(frac=1).reset_index(drop=True)
    return out_df


def create_AV_train_valid_test_sets(fp_train, 
                                    save_dir,
                                    fp_test=None,
                                    author_col="author", 
                                    text_col="text", 
                                    num_of_samples_train=100000, 
                                    num_of_samples_test=20000,
                                    valid_set_ratio=0.2,
                                    pos_ratio=0.4):

    if os.path.exists(save_dir):
        print(f"==> {save_dir} already exists. Please remove it to create a new dataset.")
        return

    print(f"===== Creating AV dataset for {save_dir} =====")

    df_train = pd.read_csv(fp_train)

    if fp_test is None:
        df_test = pd.read_csv(fp_train.replace("train", "test"))
    else:
        df_test = pd.read_csv(fp_test)

    train_df = construct_AV_dataset(df_train, author_col, text_col, 
                                    num_of_samples_train, pos_ratio)

    train_df, valid_df = train_test_split(train_df, test_size=valid_set_ratio, random_state=42)
    test_df = construct_AV_dataset(df_test, author_col, text_col, 
                                    num_of_samples_test, pos_ratio)

    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(save_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
    print(f"==> Train set size: {len(train_df)}")
    print("==> Train set label distribution:\n", train_df.label.value_counts(), "\n\n")

    print(f"==> Validation set size: {len(valid_df)}")
    print("==> Validation set label distribution:\n", valid_df.label.value_counts(), "\n\n")

    print(f"==> Test set size: {len(test_df)}")
    print("+=> Test set label distribution:\n", test_df.label.value_counts())


if __name__ == "__main__":
    
    create_AV_train_valid_test_sets("../dataset_prepare/blog_train.csv", "../dataset_prepare/blog_AV_datasets")
    create_AV_train_valid_test_sets("../dataset_prepare/CCAT50_train.csv", "../dataset_prepare/CCAT50_AV_datasets")
    create_AV_train_valid_test_sets("../dataset_prepare/enron_train.csv", "../dataset_prepare/enron_AV_datasets")
    create_AV_train_valid_test_sets("../dataset_prepare/reddit_train.csv", "../dataset_prepare/reddit_AV_datasets")
