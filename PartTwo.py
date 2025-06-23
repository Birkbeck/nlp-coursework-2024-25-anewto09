import pathlib
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
import tqdm

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

def try_vectoriser(vectoriser, print_f1_macroavg: bool = False):
    vec_train = vectoriser.fit_transform(text_train)
    vec_test = vectoriser.transform(text_test)

    classifiers = ((RandomForestClassifier(n_estimators=300), "RandomForest"),
                (LinearSVC(), "SVM with linear kernel"))
    for classifier, name in classifiers:
        classifier.fit(vec_train, party_train)
        party_pred = classifier.predict(vec_test)
        f1sc = f1_score(party_test, party_pred, average="macro")
        if print_f1_macroavg: # only print macro-avg f1 for part (c)
            print(f"{name} macro-average f1 score:", f1sc)
        print(f"{name} classification report:")
        print(classification_report(party_test, party_pred))

NEGATIVE_WORDS = {"not", "no", "never", "neither", "none", "zero", "non", "doesn't", "don't", "won't", "hasn't",
                  "hadn't", "isn't", "aren't", "ain't", "wasn't", "weren't", "can't", "shan't", "mustn't", "couldn't",
                  "shouldn't", "wouldn't", "unable", "unwilling", "incompetent", "fewer", "less", "bad", "awful",
                  "terrible", "dreadful", "evil", "disastrous"}
STOPWORDS = stopwords.words("english")
def custom_tokeniser(text: str, constituency_subs: dict[str, re.Pattern] = {}) -> list[str]:
    # replace constituency names with special tokens
    for party, pattern in constituency_subs.items():
        text = pattern.sub(party + "SAFESEAT", text)

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
    Gets the seats held by each party at any point.
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
    Produces regex patterns for replacing the names of a party's safe seats with a token identifying them as such.
    """
    # look for seats that have been held by multiple parties and remove them
    exclude = set()
    checked = set()
    for party1 in party_seats:
        checked.add(party1)
        for party2 in party_seats:
            if party2 in checked:
                continue
            exclude.update(party_seats[party1] & party_seats[party2])
        party_seats[party1].difference_update(exclude)
    # turn the constituency names into regular expressions
    return {
        party: re.compile("|".join(
            constituency for constituency in party_seats[party]
        ), re.IGNORECASE) for party in party_seats
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
    #print("Unigrams only:")
    #try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000), True)

    # part (d)
    #print("Unigrams, bigrams, and trigrams:")
    #try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3)))

    # part (e)
    print("Custom tokeniser:")

    # see docstring of get_party_seats for why using the whole dataset is ok here
    constituency_subs = get_constituency_substitutions(get_party_seats(df))

    prog = tqdm.tqdm(total=df.shape[0])
    def wrapped_tokeniser(text: str) -> list[str]:
        tokens = custom_tokeniser(text, constituency_subs)
        if len(text) >= 1000:  # need this because TfidVectorizer runs the tokenisation on the stop words
            prog.update(1)
        if prog.n == prog.total:
            prog.close()
        return tokens

    try_vectoriser(TfidfVectorizer(max_features=3000, tokenizer=wrapped_tokeniser, ngram_range=(1, 3)))