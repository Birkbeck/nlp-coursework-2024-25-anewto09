import nltk
import nltk.corpus
import spacy
import pandas as pd
import glob
import re
import os
import math

from collections import Counter
from pathlib import Path
from tqdm import tqdm # for a loading bar to give a sense of progress for slow computations

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = nltk.sent_tokenize(text)
    words = [t for s in sentences for t in no_punct_tokenise(s)]
    pronounceable_words = [w for w in words if w in d]
    num_syllables = sum(count_syl(w, d) for w in pronounceable_words)
    # Flesch-Kincaid grade formula from week 3 slides,
    # modified slightly to take into account some words not being in `d`.
    # Since the two terms can be thought of as average sentence length and average syllables in a word,
    # it seemed sensible to include words with unknown pronunciations for the former but not for the latter.
    return 0.39 *(len(words) / len(sentences)) + 11.8 * (num_syllables / len(pronounceable_words)) - 15.59

def flesch_kincaid(df: pd.DataFrame) -> dict[str, float]:
    """
    Produces a dict mapping titles to Flesch-Kincaid reading grade level scores
    """
    d = nltk.corpus.cmudict.dict()
    return {
        row["title"]: fk_level(row["text"], d) for _, row in tqdm(df.iterrows())
    }

VOWELS = {"A", "E", "I", "O", "U"}
def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    # according to the nltk documentation (https://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html)
    # the CMU pronouncing dictionary maps lower case words to lists of pronunciations.
    # Each pronunciation is a list of phonemes.
    # The way each phoneme is represented is such that the representations of vowel phonemes are start with vowels,
    # and the representations of consonant phonemes all start with consonants.
    # This is how vowel phonemes and hence number of syllables will be counted
    pronunciation = d[word][0]
    return sum(phon[0] in VOWELS for phon in pronunciation)

# tests
# d = nltk.corpus.cmudict.dict()
# print(count_syl("antidisestablishmentarianism", d), count_syl("potato", d))

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    files = glob.glob(str(path / "*.txt"))
    rows = []
    for filename in tqdm(files):
        # extract metadata from filename
        title, author, year = Path(filename).stem.split("-")
        title = title.replace("_", " ") # underscores are represented by spaces in the file names
        # extract text
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
        # save row to be put into dataframe
        rows.append((text, title, author, year))
    
    df = pd.DataFrame(rows, columns=["text", "title", "author", "year"])
    return df.sort_values('year', ignore_index=True)

def parse(df: pd.DataFrame, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    df["doc"] = [nlp(row["text"]) for _, row in tqdm(df.iterrows())]
    os.makedirs(store_path, exist_ok=True)
    pd.to_pickle(df, store_path / out_name)

def ispunctuation(token: str) -> bool:
    """Returns True if the given token is punctuation"""
    return re.match(r"^\W*$", token)

def strip_punctuation(token: str) -> str:
    """Removes punctuation from the ends of tokens, because nltk's word_tokenizer sometimes leaves punctuation affixed to words."""
    lstripped = re.sub(r"^\W+", "", token)
    return re.sub(r"\W+$", "", lstripped)

# tests
#print(strip_punctuation("fish-"), strip_punctuation("-mongers"), strip_punctuation("end."))

def no_punct_tokenise(text: str) -> list[str]:
    """Tokenises text case-insensitively without punctuation, using nltk's word tokeniser"""
    return [strip_punctuation(t).lower() for t in nltk.word_tokenize(text) if not ispunctuation(t)] 

def single_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = no_punct_tokenise(text)
    types = Counter(tokens)
    return (
        len(types) # number of token types
        /
        types.total() # number of (non-punctuation) tokens
    )

def nltk_ttr(df: pd.DataFrame) -> dict[str, float]:
    """
    Produces a dict mapping titles to type-to-token ratios
    """
    return {
        row["title"]: single_ttr(row["text"]) for _, row in tqdm(df.iterrows())
    }

def objects_counts(doc, n: int = 10) -> list[tuple[str, int]]:
    """Extracts the most common syntactic objects (as lemmas) in a parsed document. Returns a list of tuples."""
    counter = Counter(token.lemma_ for token in doc if token.dep_ == "dobj")
    return counter.most_common(n)

def subjects_by_verb_pmi(doc, target_verb, n: int = 10) -> list[str]:
    """Extracts the most common subjects of a given verb in a parsed document, by PMI. Returns a list."""
    # total number of words
    word_count = sum(1 for token in doc)
    # count of the verb. only includes occurences *as a verb*.
    verb_count = sum(token.lemma_ == target_verb and token.pos_ == "VERB" for token in doc)
    # counts of all verb subjects
    all_subj_counter = Counter(token.lemma_ for token in doc if token.dep_ == "nsubj")
    # counts of subjects of the target_verb (i.e. 'hear', in practice)
    target_verb_subj_counter = Counter(token.lemma_ for token in doc if token.dep_ == "nsubj" and token.head.lemma_ == target_verb)

    def PMI(word: str) -> float:
        return math.log(
            (target_verb_subj_counter[word] / word_count)
            /
            ((all_subj_counter[word] / word_count) * (verb_count/ word_count))
        )

    # sorted list (in order of descending PMI). We only need to look at those that actually appear with the target verb as all others have a PMI of -inf
    return list(sorted(target_verb_subj_counter.keys(), key=lambda w: PMI(w), reverse=True))[:n]

def subjects_by_verb_count(doc, verb, n: int = 10):
    """Extracts the most common subjects (as lemmas) of a given verb in a parsed document, by frequency. Returns a list."""
    counter = Counter(token.lemma_ for token in doc if token.dep_ == "nsubj" and token.head.lemma_ == verb)
    return counter.most_common(n)


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    nltk.download('punkt') # nltk insisted on this in an error message
    nltk.download('punkt_tab') # and this
    nltk.download("cmudict")

    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    print(nltk_ttr(df))
    print(flesch_kincaid(df))
    parse(df)
    print(df.head())
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print(df.head())
    
    for _, row in df.iterrows():
        print(row["title"])
        print(objects_counts(row["doc"]))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["doc"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["doc"], "hear"))
        print("\n")
