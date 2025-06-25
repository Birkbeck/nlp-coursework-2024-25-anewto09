import nltk
import pandas as pd
import pathlib
import re
import tqdm  # for a loading bar to give a sense of progress for slow computations
import trrex  # needed to optimise regex for the custom tokeniser, otherwise it takes about 6 times longer

nltk.download("stopwords")

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

DATASET_PATH = pathlib.Path(__file__).parent / "p2-texts" / "hansard40000.csv"

def clean_hansard(df: pd.DataFrame, n_parties: int = 4) -> pd.DataFrame:
    # 2(a) i.
    df = df.replace(to_replace="Labour (Co-op)",
                    value={
                        "party": "Labour"
                    })
    # 2(a) ii.
    df = df[df["party"] != "Speaker"]
    most_common_parties = set(df["party"]
                               .value_counts()
                               .sort_values(ascending=False)
                               .index[:n_parties])
    df = df[df["party"].isin(most_common_parties)]

    # 2(a) iii. and iv.
    df = df[(df["speech_class"] == "Speech") & (df["speech"].str.len() >= 1000)]
    return df

def try_vectoriser(vectoriser, print_f1_macroavg: bool = False, best_only: bool = False):
    vec_train = vectoriser.fit_transform(text_train)
    vec_test = vectoriser.transform(text_test)

    classifiers = (RandomForestClassifier(n_estimators=300), LinearSVC())
    def train_and_test(classifier):
        classifier.fit(vec_train, party_train)
        party_pred = classifier.predict(vec_test)
        return (classifier.__class__.__name__,
                f1_score(party_test, party_pred, average="macro"),
                classification_report(party_test, party_pred))
    results = [train_and_test(c) for c in classifiers]
    if best_only:
        results = [max(results, key=lambda r: r[1])]
    for res in results:
        print(res[0])
        if print_f1_macroavg:
            print("Macro-average f1 score:", res[1])
        print("Classification report:")
        print(res[2])


NEGATIVE_WORDS = {"not", "no", "never", "neither", "none", "zero", "non", "doesn't", "don't", "won't", "hasn't",
                  "hadn't", "isn't", "aren't", "ain't", "wasn't", "weren't", "can't", "shan't", "mustn't", "couldn't",
                  "shouldn't", "wouldn't", "unable", "unwilling", "incompetent", "fewer", "less", "bad", "awful",
                  "terrible", "dreadful", "evil", "disastrous"}
STOPWORDS = stopwords.words("english")
def custom_tokeniser(text: str, constituency_subs: dict[str, re.Pattern] = {}) -> list[str]:
    # replace constituency names with special tokens
    for party_token, pattern in constituency_subs.items():
        text = pattern.sub(party_token, text)

    # split on non-alphanumeric characters, except apostrophes and hyphens, keeping the separators
    tokens = re.split(r"([^\w'-]+)", text)

    # separate into actual tokens and separators
    separators = tokens[1::2]
    tokens = tokens[::2]

    # lump tokens of the same base form together, filter stopwords,
    # and append _NEG to any token between a negative word and the next punctuation,
    new_tokens = []
    neg = False
    for i, t in enumerate(tokens):
        t.lower()
        if t not in STOPWORDS:
            new_tokens.append(t + ("_NEG" if neg else ""))
        if t in NEGATIVE_WORDS:
            neg = True
        if i < len(separators) and separators[i] != " ":
            neg = False

    return new_tokens

def get_party_seats(df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Gets the seats held by each party.
    For convenience this retrieved from the Hansard dataset but in principle in could be obtained from another source,
    so the fact that part of this info may be taken from test data for the classifiers isn't really a problem.
    """
    party_seats = {}
    for _, row in df.iterrows():
        party, constituency = row["party"], row["constituency"]
        if not isinstance(constituency, str):  # some entries are numeric; ignore these
            continue
        party_seats.setdefault(party, set()).add(constituency)
    return party_seats

def get_constituency_substitutions(party_seats: dict[str, set[str]]) -> dict[str, re.Pattern]:
    """
    Produces regex patterns for replacing the names of a party's constituencies with a token identifying them as such.
    """
    # look for seats that have been held by multiple parties and remove them
    # (actually it turns out there are no such seats in the dataset, but I did not know this originally)
    exclude = set()
    checked = set()
    for party1 in party_seats:
        checked.add(party1)
        for party2 in party_seats:
            if party2 in checked:
                continue
            exclude.update(party_seats[party1] & party_seats[party2])
        party_seats[party1].difference_update(exclude)
    # turn party names into special tokens (removing any non-alphanumeric chars so that the token doesn't get split up later)
    # and turn the constituency names into regular expressions matching *any* of the party's safe seats
    return {
        re.sub(r"\W+", "", party) + "SEAT": re.compile(trrex.make(party_seats[party]), re.IGNORECASE) for party in party_seats
    }

if __name__ == "__main__":
    # part (a)
    df = pd.read_csv(DATASET_PATH)
    df = clean_hansard(df)
    print(f"Shape of cleaned Hansard dataframe: {df.shape}")

    # part (b)
    text_train, text_test, party_train, party_test = train_test_split(
        df["speech"], df["party"],
        test_size=0.5,
        random_state=26,
        shuffle=True,
        stratify=df["party"]
    )
    
    # part (b) / (c)
    print("Unigrams only:")
    try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000), print_f1_macroavg=True)

    # part (d)
    print("Unigrams, bigrams, and trigrams:")
    try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3)))

    # part (e)
    print("Custom tokeniser:")

    # see docstring of get_party_seats for why using the whole dataset is ok here
    constituency_subs = get_constituency_substitutions(get_party_seats(df))

    prog = tqdm.tqdm(total=df.shape[0])
    def wrapped_tokeniser(text: str) -> list[str]:
        tokens = custom_tokeniser(text, constituency_subs)
        prog.update(1)
        if prog.n == prog.total:
            prog.close()
        return tokens

    try_vectoriser(TfidfVectorizer(max_features=3000, tokenizer=wrapped_tokeniser, ngram_range=(1, 3)), best_only=True)
