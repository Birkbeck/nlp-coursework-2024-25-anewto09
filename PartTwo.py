import pathlib
import pandas as pd

DATASET_PATH = pathlib.Path(__file__).parent / "p2-texts" / "hansard40000.csv"
if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    print(df.head())
    print(df.info())
