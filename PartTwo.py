import pathlib
import pandas as pd
import re
from nltk.corpus import wordnet
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

def normalise_synonyms(token: str, syn_map: dict[str, str]) -> str:
    """Converts a token such that when this function is applied to all (initial) tokens, synonyms end up as the same token"""
    # first, ignore case and morphology
    token = token.lower()
    token = wordnet.morphy(token) or token 

    # if already have established 'normalised' form then can return this
    if token in syn_map:
        return syn_map[token]

    # otherwise we will make this the 'normalised' form for this word and its synonyms
    synonyms = wordnet.synonyms(token)
    
    if not synonyms:
        return token
    
    synonyms = {syn for syn_list in synonyms for syn in syn_list}
    synonyms.add(token)
    syn_map.update({syn: token for syn in synonyms})

    return token

def custom_tokeniser(text: str) -> list[str]:
    # split on non-alphanumeric characters, except apostrophes
    tokens = re.split(r"[^\w']+", text)
    # lump synonyms into same token
    syn_map = {}
    tokens = [normalise_synonyms(t, syn_map) for t in tokens]
    return tokens

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
    # print("Unigrams only:")
    # try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000), True)

    # # part (d)
    # print("Unigrams, bigrams, and trigrams:")
    # try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3)))

    # part (e)
    print("Custom tokeniser:")
    try_vectoriser(TfidfVectorizer(stop_words='english', max_features=3000, tokenizer=custom_tokeniser))