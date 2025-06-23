import pathlib
import pandas as pd
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

def try_vectoriser(vectoriser_params: dict, print_f1_macroavg: bool = False):
    vectoriser = TfidfVectorizer(**params)
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

    try_vectoriser({'stop_words': 'english', 'max_features': 3000}, True)
