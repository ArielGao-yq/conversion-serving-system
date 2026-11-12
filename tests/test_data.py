import pandas as pd

TRAIN_INDEX = "data/splits/v1/train_index.csv"
TEST_INDEX = "data/splits/v1/test_index.csv"
RAW_DATA = "data/bank-additional-full.csv"


def load_split():
    train_ids = pd.read_csv(TRAIN_INDEX)["row_id"]
    test_ids = pd.read_csv(TEST_INDEX)["row_id"]
    return train_ids, test_ids


def test_split_sizes_match_frozen_v1():
    train_ids, test_ids = load_split()
    assert len(train_ids) == 32950
    assert len(test_ids) == 8238

def test_split_has_no_overlap_and_covers_all_rows():
    train_ids, test_ids = load_split()
    train_set = set(train_ids)
    test_set = set(test_ids)
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(range(41188))


def test_split_is_temporal():
    train_ids, test_ids = load_split()
    assert train_ids.max() < test_ids.min()


def test_raw_data_columns_and_target():
    df = pd.read_csv(RAW_DATA, sep=";")
    assert len(df) == 41188
    assert "duration" in df.columns
    assert set(df["y"].unique()) == {"yes", "no"}