# example run 
# python create_stylometry_features.py --dataset blog --setting 1 --llm_name gpt4

# install the following
# pip install liwc
# pip install spacy
# python -m spacy download en_core_web_sm

import sys
import argparse
from collections import defaultdict, Counter
from statistics import stdev
import pandas as pd
import codecs
import re
import os
from tqdm import tqdm
tqdm.pandas()
import random
import csv
import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')
import re
import nltk
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
import liwc

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
import spacy
nlp = spacy.load("en_core_web_sm")

parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def get_pos_tag_counts(text):
    """
    Calculates the counts of Part-of-Speech tags in the text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        dict: A dictionary where keys are POS tags and values are their counts.
    """
    pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    pos_tag_counts = Counter({tag: 0 for tag in pos_tags})
    try:
        doc = nlp(text)
        for token in doc:
            pos_tag_counts[token.pos_] += 1
        return dict(pos_tag_counts)
    except Exception:
        return dict(pos_tag_counts)

def extract_writeprint_features(text):
    """
    Extracts a variety of linguistic features from the input text.

    Args:
        text (str): The input text.

    Returns:
        dict: A dictionary containing the extracted features and their values
              (rounded to 4 decimal points).
    """
    features = {}
    text = text.lower()
    total_chars = len(text)
    words = word_tokenize(text.lower())
    total_words = len(words)

    # 1. Letter Frequencies
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for letter in alphabet:
        count = text.lower().count(letter)
        features[f'letter_{letter}'] = round(count / total_chars, 4) if total_chars > 0 else 0.0

    # 2. Digit Frequencies
    digits = '0123456789'
    for digit in digits:
        count = text.count(digit)
        features[f'digit_{digit}'] = round(count / total_chars, 4) if total_chars > 0 else 0.0

    # 3. Special Character Frequencies (Common English punctuation and symbols)
    special_chars = '~@#$%^&*_-=+|<>[]\{\}\/\'\"\\\:'
    for char in special_chars:
        count = text.count(char)
        features[f'special_char_{re.sub(r"[^a-zA-Z0-9_]", "_", char)}'] = round(count / total_chars, 4) if total_chars > 0 else 0.0

    # 4. Bigram Frequencies (of letters)
    bigrams_list = list(ngrams(''.join(filter(str.isalpha, text.lower())), 2))
    total_bigrams = len(bigrams_list)
    common_bigrams = ['th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd', 'ed', 'or',
                       'es', 'ti', 'te', 'it', 'is', 'st', 'to', 'ar', 'of', 'ng', 'ha', 'al',
                       'ou', 'nt', 'as', 'hi', 'se', 'le', 've', 'me', 'co', 'ne', 'de', 'ea',
                       'ro', 'io', 'ri']
    bigram_counts = Counter(bigrams_list)
    for bigram in common_bigrams:
        count = bigram_counts.get(tuple(bigram), 0)
        features[f'bigram_{bigram}'] = round(count / total_bigrams, 4) if total_bigrams > 0 else 0.0

    # 5. Trigram Frequencies (of letters)
    trigrams_list = list(ngrams(''.join(filter(str.isalpha, text.lower())), 3))
    total_trigrams = len(trigrams_list)
    common_trigrams = ['the', 'and', 'ing', 'ion', 'ent', 'tio', 'her', 'for', 'hat', 'tha',
                        'his', 'ter', 'ere', 'ati', 'ate', 'was', 'all', 'ver', 'ith', 'thi']
    trigram_counts = Counter(trigrams_list)
    for trigram in common_trigrams:
        count = trigram_counts.get(tuple(trigram), 0)
        features[f'trigram_{trigram}'] = round(count / total_trigrams, 4) if total_trigrams > 0 else 0.0

    # 6. Function Word Frequencies (a predefined list of common function words)
    function_words_list = [
        'a', "he's", 'since', 'about', 'highly', 'above', 'him', 'absolutely', 'himself', 'so',
        'across', 'his', 'some', 'actually', 'hopefully', 'somebody', 'after', 'how', 'somehow',
        'again', 'however', 'someone', 'against', 'hundred', 'something', 'ahead', 'i', 'somewhat',
        'somewhere', "ain't", "i'd", 'soon', 'all', 'if', "i'll", 'still', 'along', 'stuff',
        'alot', "i'm", 'such', 'also', 'immediately', 'ten', 'although', 'in', 'tenth', 'am',
        'infinity', 'than', 'among', 'inside', 'that', 'an', 'insides', 'thatd', 'and', 'instead',
        "that'd", 'another', 'into', 'any', 'is', "that'll", 'anybody', 'item', 'thats',
        'anymore', "isn't", "that's", 'anyone', 'it', 'the', 'anything', "it'd", 'thee',
        'anyway', 'their', 'anywhere', "it'll", 'them', 'apparently', 'themselves', 'are',
        "it's", 'then', "aren't", 'its', 'there', 'around', 'itself', "there's", 'as', 'these',
        'at', "i've", 'they', 'atop', 'just', "they'd", 'away', 'lack', "they'll", 'back',
        'lately', "they're", 'basically', 'least', "they've", 'be', 'less', 'thing', 'became',
        'let', 'third', 'become', "let's", 'thirty', 'becomes', 'loads', 'this', 'becoming',
        'lot', 'those', 'been', 'lots', 'thou', 'before', 'main', 'though', 'behind', 'many',
        'thousand', 'being', 'may', 'three', 'below', 'maybe', 'through', 'beneath', 'me', 'thru',
        'beside', 'might', 'thy', 'besides', "might've", 'till', 'best', 'million', 'to',
        'between', 'mine', 'ton', 'beyond', 'more', 'tons', 'billion', 'most', 'too', 'both',
        'mostly', 'total', 'bunch', 'much', 'totally', 'but', 'must', 'toward', 'by', "must'nt",
        'trillion', 'can', "mustn't", 'truly', 'cannot', "must've", 'twice', "can't", 'my',
        'two', 'clearly', 'myself', "need'nt", 'under', 'completely', 'near', 'underneath',
        'constantly', 'nearly', "needn't", 'unique', 'could', 'neither', 'unless', "couldn't",
        'never', 'until', "could've", 'nine', 'unto', 'couple', 'no', 'up', 'cuz', 'nobody',
        'upon', 'definitely', 'none', 'us', 'despite', 'nope', 'usually', 'did', 'nor', 'various',
        "didn't", 'not', 'very', 'difference', 'nothing', 'wanna', 'do', 'now', 'was', 'does',
        'nowhere', 'wasnt', 'doesnt', 'of', "wasn't", "don't", 'off', 'we', 'doubl', 'often',
        "we'd", 'down', 'on', "we'll", 'dozen', 'once', 'were', 'during', 'one', "we're", 'each',
        'ones', "weren't", 'eight', 'oneself', "we've", 'either', 'only', 'what', 'eleven', 'onto',
        'whatever', 'else', 'or', "what's", 'enough', 'other', 'when', 'entire', 'others',
        'whenever', 'equal', 'otherwise', 'where', 'especially', 'ought', 'whereas', 'etc',
        'oughta', "where's", 'even', "ought'nt", 'whether', 'eventually', "oughtn't", 'which',
        'ever', 'oughtve', 'whichever', 'every', "ought've", 'while', 'everybod', 'our', 'who',
        'everyone', 'ours', "who'd", 'everything', 'ourselves', "who'll", 'example', 'out', 'whom',
        'except', 'outside', 'whose', 'extra', 'over', 'will', 'extremely', 'own', 'with', 'fairly',
        'part', 'within', 'few', 'partly', 'without', 'fift', 'perhaps', 'wont', 'first', 'piece',
        "won't", 'firstly', 'plenty', 'worst', 'firsts', 'plus', 'would', 'five', 'primarily',
        "wouldn't", 'for', 'probably', "would've", 'form', 'quarter', 'yours', 'four', 'quick',
        'zero', 'frequently', 'rarely', 'zillion', 'from', 'rather',
        'full', 'really', 'generally', 'remaining', 'greater', 'rest', 'greatest', 'same', 'had',
        'second', "hadn't", 'section', 'half', 'seriously', 'has', 'seven', 'shall', "hasn't",
        'several', "shan't", 'have', 'she', 'you', 'havent', "she'd", "you'd", "haven't", "she'll",
        "you'll", 'having', "she's", 'your', 'he', 'should', "you're", "he'd", "should'nt",
        "you've", "he'll", "shouldn't", 'herself', "should've", "he's", 'her', 'so', 'here',
        'some', "here's", 'somebody', 'hers', 'somehow', 'herself', 'someone', 'his', 'something',
        'how', 'somewhat', 'i', 'somewhere', 'i', 'soon', "i'd", 'still', "i'll", 'such', "i'm",
        'ten', "i've", 'tenth', 'that', 'than', "that'd", 'the', "that'll", 'thee', "that's",
        'their', 'them', 'then', 'themselves', 'there', 'these', "there's", 'they', 'they',
        "they'd", "they'll", "they're", "they've", 'thing', 'third', 'thirty', 'this', 'those',
        'thou', 'though', 'thousand', 'three', 'thru', 'thy', 'till', 'to', 'ton', 'tons', 'too',
        'total', 'totally', 'toward', 'trillion', 'truly', 'twice', 'two', 'under', 'underneath',
        'unique', 'unless', 'until', 'unto', 'up', 'upon', 'us', 'usually', 'various', 'very',
        'wanna', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "weren't", "we've", 'what',
        "what's", 'when', 'whenever', 'where', "where's", 'whereas', 'whether', 'which',
        'whichever', 'while', 'who', "who'd", "who'll", 'whom', 'whose', 'why', 'will', 'with',
        'within', 'without', 'wont', "won't", 'would', "wouldn't", "would've", 'yall', "y'all",
        'you', "you'd", "you'll", 'your', "you're", 'yours', "you've", 'zero', 'zillion'
    ]
    word_counts = Counter(words)
    total_relevant_words = sum(word_counts.get(fw, 0) for fw in function_words_list)
    for fw in function_words_list:
        features[f'function_word_{fw}'] = round(word_counts.get(fw, 0) / total_words, 4) if total_words > 0 else 0.0

    # 7. Part-of-Speech (POS) Tag Frequencies (using spaCy)
    pos_tag_counts = get_pos_tag_counts(text)
    total_pos = sum(pos_tag_counts.values())
    common_pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                  'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    for pos in common_pos:
        features[f'pos_{pos}'] = round(pos_tag_counts.get(pos, 0) / total_pos, 4) if total_pos > 0 else 0.0
    if not total_pos > 0:
        for pos in common_pos:
            features[f'pos_{pos}'] = 0.0

    # 8. Lexical Diversity (Hapax Legomena and Dis Legomena Ratios)
    unique_words = set(words)
    word_counts_all = Counter(words)
    hapax_legomena = sum(1 for word, count in word_counts_all.items() if count == 1)
    dis_legomena = sum(1 for word, count in word_counts_all.items() if count == 2)

    features['hapax_legomena_ratio'] = round(hapax_legomena / total_words, 4) if total_words > 0 else 0.0
    features['dis_legomena_ratio'] = round(dis_legomena / total_words, 4) if total_words > 0 else 0.0

    # 9. Average Word Length
    total_word_length = sum(len(word) for word in words)
    features['avg_word_length'] = round(total_word_length / total_words, 4) if total_words > 0 else 0.0

    # 10. Ratio of Short Words (length <= 3)
    short_word_count = sum(1 for word in words if len(word) <= 3)
    features['short_words'] = round(short_word_count / total_words, 4) if total_words > 0 else 0.0

    # 11. Ratio of Digits within Words
    digits_in_words_count = sum(sum(1 for char in word if char.isdigit()) for word in words)
    total_chars_in_words = sum(len(word) for word in words)
    features['digits_ratio'] = round(digits_in_words_count / total_chars_in_words, 4) if total_chars_in_words > 0 else 0.0

    return features

def get_liwc_features(text):
  text = text.lower()
  tokenlist = tokenize(text)
  wc = len(tokenlist)
  # print(wc)

  STYLE_FEATURES = {}
  token_counts = Counter(category for token in tokenlist for category in parse(token))

  for category in category_names:
    f = 'liwc_'+ category + '_frac'
    if category not in token_counts.keys():  # then frequency 0
      STYLE_FEATURES[f] = 0.0
    else:
      STYLE_FEATURES[f] = round(token_counts[category] / wc, 4) if wc > 0 else 0.0
  return  STYLE_FEATURES



def get_args():
    parser = argparse.ArgumentParser(description="Extract stylometry features from LLM writing samples.")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., 'blog', 'reddit', etc.")
    parser.add_argument("--setting", type=int, choices=[1, 2, 3, 4, 5], required=True, help="Prompt setting (1-5).")
    parser.add_argument("--llm_name", type=str, required=True, help="LLM model name, e.g., 'gpt4', 'gpt3.5', etc.")

    return parser.parse_args()

# -----------------------------------
# Main Function
# -----------------------------------
def main():
    args = get_args()
    dataset = args.dataset
    setting = args.setting
    llm_name = args.llm_name

    input_filename = f'LLM_writing//Setting{setting}/{dataset}/{llm_name}.csv'
    if os.path.exists(input_filename):
      df = pd.read_csv(input_filename)
    else:
      print(f"File '{input_filename}' does not exist.")
      sys.exit(1)

    print(df.shape)

    # Output path
    output_filename = f'Style_features_LLM/Setting{setting}/{dataset}/{llm_name}_features.csv'
    output_dir = os.path.dirname(output_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    else:
        print(f"✅ Output directory already exists: {output_dir}")

    if os.path.exists(output_filename):
      print(f"File '{output_filename}' already exists.")
      sys.exit(1)

    print('processing liwc features')
    df['liwc_features'] = df['writing'].progress_apply(lambda x: get_liwc_features(x) if isinstance(x, str) else {})
    print('processing writeprint features')
    df['writeprint_features'] = df['writing'].progress_apply(lambda x: extract_writeprint_features(x) if isinstance(x, str) else {})
    df.to_csv(output_filename, index=False)

    print(f'stylometry feature extraction completed for Setting{setting}/{dataset}/{llm_name}')


# -----------------------------------
# Entry point
# -----------------------------------
if __name__ == "__main__":
    main()