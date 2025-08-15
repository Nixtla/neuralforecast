import pandas as pd


 # Set up column renaming for Polars
def assert_equal_dfs(pandas_df, polars_df):
    """Helper function to compare Pandas and Polars dataframes."""
    renamer = {"unique_id": "uid", "ds": "time", "y": "target"}
    inverse_renamer = {v: k for k, v in renamer.items()}
    mapping = {k: v for k, v in inverse_renamer.items() if k in polars_df}
    pd.testing.assert_frame_equal(
        pandas_df,
        polars_df.rename(mapping).to_pandas(),
    )

def get_expected_size(df, h, test_size, step_size):
    expected_size = 0
    uids = df["unique_id"].unique()
    for uid in uids:
        input_len = len(df[df["unique_id"] == uid])
        expected_size += ((input_len - test_size - h) / step_size + 1) * h
    return expected_size
