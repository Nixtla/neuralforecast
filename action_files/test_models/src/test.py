import pandas as pd


def main() -> None:
    Y_df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/M3-Monthly.csv', parse_dates=['ds']).head()
    Y_df.to_csv('data/evaluation.csv')

if __name__ == '__main__':
    main()