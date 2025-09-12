Data and code for our paper *Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors*, which is accepted to EMNLP 2025 (Findings).



## Data

We used the following datasets for all the experiments in our paper: [Enron](https://www.cs.cmu.edu/~enron/) (Klimt and Yang, 2004), [Blog](https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) (Schler et al., 2006), [CCAT50](https://archive.ics.uci.edu/dataset/217/reuter+50+50) (Liu, 2011), and [Reddit](https://huggingface.co/datasets/webis/tldr-17) (V"olske et al., 2017).

To reproduce our results, please donwload the two zip files below and unzip them to the current folder folder (where the folder `scripts` is) . 

### [dataset_prepare](https://drive.google.com/file/d/1zzlnY_OTr6Mtnt19DQBtYF5LttF6DOt2/view?usp=sharing): Main Experiments

This zip file contains

- the train and test splits for the four datasets (Enron, Blog, CCAT50, Reddit), totalling eight csv files:
  - The train sets are used for training AA (authorship attribution) models.
  - The test sets come with a content summary for each writing sample.

- and four folders containing **constructed** AV (authorship verification) datasets for the four datasets. 
  - Each folder comprises three csv files (train.csv, valid.csv, and test.csv) for training and evaluating an AV model on the original human-authored data. 
  - We use the `create_AV_datasets.py` inside the `scripts` folder to construct these AV datasets.  



### [dataset_followup](https://drive.google.com/file/d/1fxkoClxjfkrUpzqy9rfPZk7tcCobkQG1/view?usp=sharing): Follow-up Experiments (Section 6)

This zip file contains subsets of the original test splits used for the follow-up experiments in addition to the original train splits (cluster labelled) for the four datasets.  



## Code 

To run our code, please create an virtual environment and install the following dependencies:

1. Create a conda environment: `conda create -n PW python=3.12.9`
2. Use the `PW` env whenever running the code: `conda activate PW`
3. install pip
4. run `pip install requirements.txt`



#### LLM Writing Generation 

- `generate_llm_writing.py`: The script includes all the LLM generation experiments conducted in the paper. We use [litellm](https://github.com/BerriAI/litellm) when making all of our LLM API calls. 
- Example command line:

```
> setting=1
> LLM="openai/gpt-4o-mini-2024-07-18"
> python generate_llm_writing.py --training_df_fp=dataset_prepare/blog_train.csv --evaluation_df_fp=dataset_prepare/blog_test.csv --setting=$setting --llm=$LLM
```



- There are a total of settings explained below:
  - Setting 1: few-shot prompting. Deafults to 5-shot promting (Main Experiments). 
  - Setting 2: few-shot prompting where the exemplars are from the same clsuter of the test sample. (**+Sim Ctrl**)
  - Setting 3: few-shot prompting where the exemplars are similarly long to the test sample. (**+Len Ctrl**)
  - Setting 4: 0-shot prompting. Only a content summary is given (Baselines for the Main Experiments).
  -  Setting 5: few-shot prompting where besides a content summary, a text snippet of the test sample is also included in the prompt (**+Snippet**). 
  - Setting 6: few-shot prompting for a set of numbers (defaults to 2, 4, 6, 8, 10) to examine the effect of the number of exemplars. 



#### Evaluation Metrics

- `train_and_eval_an_AA_model.py`: used for training and evaluating an AA model. 
  
  - `deploy_AA_models.py`: used for deploying the trained AA models to the LLM generated samples.  
  
  - Example command line: 

```
> python train_and_eval_an_AA_model.py --training_df_fp=dataset_prepare/enron_train.csv --test_df_fp=dataset_prepare/enron_test.csv --num_train_epochs=20 --model_name=answerdotai/ModernBERT-base
```



- `train_and_eval_an_AV_model.py`: used for training and evaluating an AV model. 
  - `deploy_AV_models.py`: used for deploying the trained AV models to the LLM generated samples. 
  - Example command line:

```
> python train_and_eval_an_AV_model.py --data_dir=dataset_prepare/CCAT50_AV_datasets --num_train_epochs=10 --model_name=allenai/longformer-base-4096 --max_length=2048
```



- `create_stylometry_features.py`: the style models 
  - developed based on [LIWC (Boyd et al., 2022)](https://www.liwc.app/static/documents/LIWC-22%20Manual%20-%20Development%20and%20Psychometrics.pdf) and [WritePrint (Abbasi and Chen, 2008)](https://dl.acm.org/doi/10.1145/1344411.1344413).
  - `LIWC2007_English100131.dic` contains the LIWC features.