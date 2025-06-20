import pathlib
import pandas as pd

DATASET_PATH = pathlib.Path(__file__).parent / "p2-texts" / "hansard40000.csv"

def clean_hansard(df: pd.DataFrame, n_parties: int = 4) -> pd.DataFrame:
    df = df.replace(to_replace="Labour",
                    value={
                        "party": "Labour"
                    })
    most_common_parties = set(df["party"]
                               .value_counts()
                               .sort_values(ascending=False)
                               .index[:n_parties])
    print(most_common_parties)


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    clean_hansard(df)
