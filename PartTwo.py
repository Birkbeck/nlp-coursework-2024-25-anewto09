import pathlib
import pandas as pd

DATASET_PATH = pathlib.Path(__file__).parent / "p2-texts" / "hansard40000.csv"

def clean_hansard(df: pd.DataFrame, n_parties: int = 4) -> pd.DataFrame:
    # 2(a) i.
    df = df.replace(to_replace="Labour",
                    value={
                        "party": "Labour"
                    })
    # 2(a) ii.
    most_common_parties = set(df["party"]
                               .value_counts()
                               .sort_values(ascending=False)
                               .index[:n_parties])
    df = df[df["party"].isin(most_common_parties)]

    return df

if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    print(df["party"].tail())
    print(clean_hansard(df)["party"].tail())
