import os
import argparse
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from scripts.utils import get_completion

from scripts.utils import (
    count_words,
    round_up_to_nearest_10,
    list_writing_samples, 
    align_df1_to_df2
)

from scripts.prompt_templates import (
 get_prompt_template_for_writing_setting1, 
 get_prompt_template_for_writing_setting2,
 get_prompt_template_for_writing_setting3,
 get_prompt_template_for_writing_setting4, 
 get_prompt_template_for_writing_setting5,   
)


def get_args():
    parser = argparse.ArgumentParser(description="Create writing prompts and prompt LLMs to generate writing.")
    parser.add_argument("--evaluation_df_fp", type=str, required=True, help="Path to the evaluation DataFrame.")
    parser.add_argument("--llm", type=str, required=True, help="LLM model to use for generation. Use litellm name convention.")

    parser.add_argument("--training_df_fp", type=str, default=None, help="Path to the training DataFrame. Default is None.")
    parser.add_argument("--setting", type=int, choices=[1, 2, 3, 4, 5, 6], default=1, help="Prompt setting (1-6). Default is 1.")
    parser.add_argument("--genre", type=str, default=None, help="Genre of the writing samples. Default is None (auto infer from dataset if possible).")
    parser.add_argument("--author_col", type=str, default="author", help="Column name for author in the DataFrame. Default is 'author'.")
    parser.add_argument("--text_col", type=str, default="text", help="Column name for text in the DataFrame. Default is 'text'.")
    parser.add_argument("--summary_col", type=str, default="summary", help="Column name for summary in the DataFrame. Default is 'summary'.")
    parser.add_argument("--num_exemplars", type=int, default=5, help="Number of exemplars per author. Default is 5.")
    parser.add_argument("--nums_exemplars", type=str, default="2,4,6,8,10", help="Comma-separated list of numbers of exemplars for Setting 6. Default is '2,4,6,8,10'.")
    
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the LLM. Default is 0.")
    parser.add_argument("--max_tries", type=int, default=5, help="Number of tries for LLM completion. Default is 5.")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving LLM outputs. Default is 10.")

    return parser.parse_args()


def create_writing_prompts_setting1(training_df_fp, 
                                    evaluation_df_fp, 
                                    genre,
                                    author_col="author", 
                                    text_col="text", 
                                    summary_col="summary", 
                                    num_exemplars=5):
    '''Create writing prompts for the evaluation set based on the training set.
    For each sample in the evaluation set, find num_exemplars random samples
    in the training set to be used as writing examples. 
    The prompt will include the writing samples and the summary of the evaluation sample.
    '''
    
    training_df = pd.read_csv(training_df_fp)
    evaluation_df = pd.read_csv(evaluation_df_fp)

    assert training_df[author_col].value_counts().min() >= num_exemplars, \
        f"Each author must have at least {num_exemplars} samples in the training set."
    
    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."

    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting1()        
    
    print(f"Generating prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        
        author = row[author_col]
        summary = row[summary_col]
        
        num_words = round_up_to_nearest_10(count_words(row[text_col]))
        samples = training_df[training_df[author_col]==author][text_col].sample(num_exemplars)
        writing_samples = list_writing_samples(samples)
        prompt = prompt_tmp.substitute(writing_samples=writing_samples, 
                                       genre=genre, num_words=num_words,
                                       summary=summary)
        evaluation_df.at[ix, "training sample indices"] = ",".join([str(ix) for ix in samples.index])
        evaluation_df.at[ix, "prompt"] = prompt
    
    return evaluation_df


def create_writing_prompts_setting2(training_df_fp, 
                                    evaluation_df_fp, 
                                    genre,
                                    author_col="author", 
                                    text_col="text", 
                                    summary_col="summary", 
                                    num_exemplars=5):
    '''Create writing prompts for the evaluation set based on the training set.
    For each sample in the evaluation set, find the num_exemplars most similar samples
    in the training set that belong to the same cluster of the evaluation sample. 
    The prompt will include the writing samples and the summary of the evaluation sample.
    '''
    
    training_df = pd.read_csv(training_df_fp)
    evaluation_df = pd.read_csv(evaluation_df_fp)

    assert training_df[author_col].value_counts().min() >= num_exemplars, \
        f"Each author must have at least {num_exemplars} samples in the training set."
    
    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."
    
    assert "cluster" in training_df.columns, \
        f"Cluster column 'cluster' not found in training DataFrame."
    
    assert "cluster" in evaluation_df.columns, \
        f"Cluster column 'cluster' not found in evaluation DataFrame."

    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting2()        
    
    print(f"Generating prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        
        author = row[author_col]
        summary = row[summary_col]
        cluster = row["cluster"]
        
        num_words = round_up_to_nearest_10(count_words(row[text_col]))
        samples = training_df[(training_df[author_col]==author) & 
                              (training_df["cluster"]==cluster)][text_col].sample(num_exemplars)
        
        writing_samples = list_writing_samples(samples)
        prompt = prompt_tmp.substitute(writing_samples=writing_samples, 
                                       genre=genre, num_words=num_words,
                                       summary=summary)
        evaluation_df.at[ix, "training sample indices"] = ",".join([str(ix) for ix in samples.index])
        evaluation_df.at[ix, "prompt"] = prompt
    
    return evaluation_df




def create_writing_prompts_setting3(training_df_fp, 
                                    evaluation_df_fp, 
                                    genre,
                                    author_col="author", 
                                    text_col="text", 
                                    summary_col="summary", 
                                    num_exemplars=5):
    '''Create writing prompts for the evaluation set based on the training set.
    For each sample in the evaluation set, find the num_exemplars most similar samples
    in the training set based on word count. The prompt will include the writing samples
    and the summary of the evaluation sample.
    '''
    
    training_df = pd.read_csv(training_df_fp)
    evaluation_df = pd.read_csv(evaluation_df_fp)
    training_df["num_words"] = training_df[text_col].apply(count_words)

    assert training_df[author_col].value_counts().min() >= num_exemplars, \
        f"Each author must have at least {num_exemplars} samples in the training set."
    
    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."

    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting3()        
    
    print(f"Generating prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        
        author = row[author_col]
        summary = row[summary_col]
        
        num_words = count_words(row[text_col])
        samples = training_df.copy()[training_df[author_col]==author]
        samples["wc_diff"] = abs(samples["num_words"] - num_words)
        samples = samples.sort_values("wc_diff", ).head(num_exemplars)
        
        writing_samples = list_writing_samples(samples)
        prompt = prompt_tmp.substitute(writing_samples=writing_samples, 
                                       genre=genre, num_words=round_up_to_nearest_10(num_words),
                                       summary=summary)
        evaluation_df.at[ix, "training sample indices"] = ",".join([str(ix) for ix in samples.index])
        evaluation_df.at[ix, "prompt"] = prompt

    
    return evaluation_df


def create_writing_prompts_setting4(evaluation_df_fp, 
                                    genre,
                                    text_col="text", 
                                    summary_col="summary"):
    '''Create writing prompts for the evaluation set based a summary.
    '''
    evaluation_df = pd.read_csv(evaluation_df_fp)

    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."
    
    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting4()
    
    print(f"Generating prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        summary = row[summary_col]
        
        num_words = round_up_to_nearest_10(count_words(row[text_col]))
        prompt = prompt_tmp.substitute(genre=genre, num_words=num_words,
                                       summary=summary)
        evaluation_df.at[ix, "prompt"] = prompt

    return evaluation_df


def create_writing_prompts_setting5(training_df_fp, 
                                    evaluation_df_fp, 
                                    genre,
                                    author_col="author", 
                                    text_col="text", 
                                    summary_col="summary", 
                                    num_exemplars=5):
    
    def get_text_snippet(text, percentage=0.2):
        words = word_tokenize(text)
        num_words = len(words)
        snippet_length = min(50, int(num_words * percentage))
        length_to_continue = num_words - snippet_length
        snippet = " ".join(words[:snippet_length])
        return snippet, length_to_continue
    
    training_df = pd.read_csv(training_df_fp)
    evaluation_df = pd.read_csv(evaluation_df_fp)

    assert training_df[author_col].value_counts().min() >= num_exemplars, \
        f"Each author must have at least {num_exemplars} samples in the training set."
    
    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."

    evaluation_df = evaluation_df.copy().reset_index(drop=True)
    summary_only_prompt_tmp, exemplars_plus_summary_prompt_tmp = \
        get_prompt_template_for_writing_setting5()

    print(f"Generating summary-only prompts...")
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        summary = row[summary_col]
        snippet, length_to_continue = get_text_snippet(row[text_col])
        length_to_continue = round_up_to_nearest_10(length_to_continue)
        prompt = summary_only_prompt_tmp.substitute(genre=genre, 
                                                    num_words=length_to_continue, 
                                                    summary=summary, 
                                                    snippet=snippet)
        evaluation_df.at[ix, "training sample indices"] = "-"
        evaluation_df.at[ix, "prompt"] = prompt

    evaluation_df["Condition"] = "summary-only"
    out = [evaluation_df.copy()]
    print(f"Generating exemplars-plus-summary prompts...")
    
    dataset = evaluation_df_fp.split("/")[-1].split("_")[0]
    earlier_setting1_prompts_df = None
    earlier_setting1_prompts_fp = f"LLM_writing/Setting1/{dataset}/prompts.csv"
    
    if os.path.exists(earlier_setting1_prompts_fp):
        earlier_setting1_prompts_df = pd.read_csv(earlier_setting1_prompts_fp)
        earlier_setting1_prompts_df = align_df1_to_df2(earlier_setting1_prompts_df, evaluation_df, 
                                                       text_col, author_col, summary_col)
        use_random_samples = True if earlier_setting1_prompts_df is None else False
    
    for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        author = row[author_col]
        summary = row[summary_col]
        snippet, length_to_continue = get_text_snippet(row[text_col])
        length_to_continue = round_up_to_nearest_10(length_to_continue)

        if earlier_setting1_prompts_df is not None and not use_random_samples:

            indices = [int(i) for i in earlier_setting1_prompts_df.at[ix, "training sample indices"].split(",")]
            samples = training_df.loc[indices]

            if len(samples) != num_exemplars:
                print(f"Number of samples found for {row[text_col]} in earlier setting1 prompts is not {num_exemplars}. ")
                use_random_samples = True
                break
                
            if len(samples[author_col].unique()) != 1:
                print(f"More than one author found for sample {row[text_col]} in earlier setting1 prompts. " \
                      f"Found authors: {samples[author_col].unique()}")
                use_random_samples = True
                break

            if samples[author_col].values[0] != author:
                print(f"Author mismatch for sample {row[text_col]} in earlier setting1 prompts.")
                use_random_samples = True
                break

            samples = samples[text_col]
        
        if use_random_samples:
            samples = training_df[training_df[author_col]==author][text_col].sample(num_exemplars)

        writing_samples = list_writing_samples(samples)
        prompt = exemplars_plus_summary_prompt_tmp.substitute(writing_samples=writing_samples, 
                                                              genre=genre, 
                                                              num_words=length_to_continue,
                                                              summary=summary, 
                                                              snippet=snippet)
        
        evaluation_df.at[ix, "training sample indices"] = ",".join([str(ix) for ix in samples.index])
        evaluation_df.at[ix, "prompt"] = prompt
    
    evaluation_df["Condition"] = "exemplars-plus-summary"
    
    out.append(evaluation_df.copy())
    out_df = pd.concat(out, ignore_index=True)
    return out_df


def create_writing_prompts_setting6(training_df_fp, 
                                    evaluation_df_fp, 
                                    genre,
                                    author_col="author", 
                                    text_col="text", 
                                    summary_col="summary", 
                                    nums_exemplars=[2, 4, 6, 8, 10]):
    '''Similar to setting 1, but for each sample in the evaluation set,
    find multiple num_exemplars random samples in the training set to be used as writing examples.
    A larger nums_exemplar will subsume all smaller nums_exemplar. 
    '''
    
    training_df = pd.read_csv(training_df_fp)
    evaluation_df = pd.read_csv(evaluation_df_fp)
    nums_exemplars = sorted(nums_exemplars, reverse=True)

    assert training_df[author_col].value_counts().min() >= nums_exemplars[0], \
        f"Each author must have at least {nums_exemplars[0]} samples in the training set."
    
    assert summary_col in evaluation_df.columns, \
        f"Summary column '{summary_col}' not found in evaluation DataFrame."

    evaluation_df = evaluation_df.copy()
    prompt_tmp = get_prompt_template_for_writing_setting1()        
    
    print(f"Generating prompts...")
    out = []
    exemplars_map = {}
    for num_exemplar in nums_exemplars:
        for ix, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
            
            author = row[author_col]
            summary = row[summary_col]
            
            num_words = round_up_to_nearest_10(count_words(row[text_col]))

            if ix in exemplars_map:
                samples = exemplars_map[ix][:num_exemplar]
            else:
                samples = training_df[training_df[author_col]==author][text_col].sample(num_exemplar)
                exemplars_map[ix] = samples
            
            writing_samples = list_writing_samples(samples)
            prompt = prompt_tmp.substitute(writing_samples=writing_samples, 
                                        genre=genre, num_words=num_words,
                                        summary=summary)
            evaluation_df.at[ix, "training sample indices"] = ",".join([str(ix) for ix in samples.index])
            evaluation_df.at[ix, "prompt"] = prompt

        out.append(evaluation_df.copy())
        out[-1]["num_exemplars"] = num_exemplar
    
    out_df = pd.concat(out, axis=0).reset_index(drop=True)
    return out_df


def generate_or_load_writing_prompts(args, dire):
    if os.path.exists(os.path.join(dire, "prompts.csv")):
        print(f"Prompts already exist in {dire}/prompts.csv")
        df = pd.read_csv(os.path.join(dire, "prompts.csv"))

        return df
    
    if args.setting == 1:

        assert args.training_df_fp is not None, \
            "Training DataFrame path is required for setting 1."
        
        df = create_writing_prompts_setting1(
            training_df_fp=args.training_df_fp,
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            author_col=args.author_col,
            text_col=args.text_col,
            summary_col=args.summary_col,
            num_exemplars=args.num_exemplars
        )

    elif args.setting == 2:
        assert args.training_df_fp is not None, \
            "Training DataFrame path is required for setting 2."
        
        df = create_writing_prompts_setting2(
            training_df_fp=args.training_df_fp,
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            author_col=args.author_col,
            text_col=args.text_col,
            summary_col=args.summary_col,
            num_exemplars=args.num_exemplars
        )
    
    elif args.setting == 3:
        assert args.training_df_fp is not None, \
            "Training DataFrame path is required for setting 3."
        
        df = create_writing_prompts_setting3(
            training_df_fp=args.training_df_fp,
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            author_col=args.author_col,
            text_col=args.text_col,
            summary_col=args.summary_col,
            num_exemplars=args.num_exemplars
        )

    elif args.setting == 4:
        df = create_writing_prompts_setting4(
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            text_col=args.text_col,
            summary_col=args.summary_col
        )
    
    elif args.setting == 5:
        assert args.training_df_fp is not None, \
            "Training DataFrame path is required for setting 5."
        
        df = create_writing_prompts_setting5(
            training_df_fp=args.training_df_fp,
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            author_col=args.author_col,
            text_col=args.text_col,
            summary_col=args.summary_col,
            num_exemplars=args.num_exemplars
        )
    
    elif args.setting == 6:
        assert args.training_df_fp is not None, \
            "Training DataFrame path is required for setting 6."
        
        nums_exemplars = [int(x) for x in args.nums_exemplars.split(",")]
        df = create_writing_prompts_setting6(
            training_df_fp=args.training_df_fp,
            evaluation_df_fp=args.evaluation_df_fp,
            genre=args.genre,
            author_col=args.author_col,
            text_col=args.text_col,
            summary_col=args.summary_col,
            nums_exemplars=nums_exemplars
        )
    else:

        raise ValueError("Setting not implemented yet.")
    
    df.to_csv(os.path.join(dire, "prompts.csv"), index=False)
    print(f"Prompts saved to {dire}/prompts.csv")

    return df


def prompt_llm_to_generate_writing(df, save_dir, model, 
                                   temperature=0, max_tries=5, 
                                   save_freq=10):
    model_name = model.split("/")[-1]
    fp = os.path.join(save_dir, model_name + ".csv")

    if os.path.exists(fp):
        llm_df = pd.read_csv(fp)
        llm_df.writing = llm_df.writing.replace("SOMETHING_WRONG", pd.NA)
        num_already_generated = llm_df.writing.notna().sum()
        if len(df) == num_already_generated:
            print(f"All writing samples already generated for {model_name}.")
            return
        else:
            print(f"{num_already_generated} writing samples generated for {model_name}. Continuing...")
            indices = sorted(set(df.index) - set(llm_df[llm_df.writing.notna()].index))
    else:
        indices = list(range(len(df)))
        llm_df = pd.DataFrame(columns=["writing"])
    
    for j, ix in tqdm(enumerate(indices), total=len(indices)):
        prompt = df.at[ix, "prompt"]
        completion = get_completion(prompt, model=model, 
                                    temperature=temperature, 
                                    max_tries=max_tries)
        llm_df.at[ix, "writing"] = completion

        if (j+1) % save_freq == 0:
            llm_df.to_csv(fp, index=False)
    
    llm_df.to_csv(fp, index=False)
    print(f"Writing generated and saved to {fp}")


def main():
    args = get_args()
    print(args)
    
    dataset = args.evaluation_df_fp.split("/")[-1].split(".")[0].split("_")[0]
    dire = f"LLM_writing/Setting{args.setting}/{dataset}"
    os.makedirs(dire, exist_ok=True)

    if args.training_df_fp is not None:
        dataset_ = args.training_df_fp.split("/")[-1].split(".")[0].split("_")[0]
        assert dataset == dataset_, \
            f"Training and evaluation datasets must be the same. {dataset} != {dataset_}"

    if args.genre is None:
        if dataset.startswith("blog"):
            args.genre = "blog post"
        elif dataset.startswith("enron"):
            args.genre = "email"
        elif dataset.startswith("reddit"):
            args.genre = "reddit post"
        elif dataset.startswith("CCAT50"):
            args.genre = "news article"
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Please specify a genre.")

    #### generating or loading prompts
    df = generate_or_load_writing_prompts(args, dire)

    #### prompting llm to generate writing
    prompt_llm_to_generate_writing(
        df, 
        save_dir=dire, 
        model=args.llm,
        temperature=args.temperature,
        max_tries=args.max_tries,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()
