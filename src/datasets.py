import pandas as pd

data_all = pd.read_csv("../data/tidy_data/data_merged.csv")
data_bp = pd.read_csv("../data/tidy_data/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_merge_ready.csv")
data_c = pd.read_csv("../data/tidy_data/2011-2020_verkehrszaehlungen_werte_auto_merge_ready.csv")
data_m = pd.read_csv("../data/tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.csv")

if __name__ == '__main__':
    pd.options.display.max_columns = 10
    pd.set_option('display.width', 200)
    print(data_all.info())
    print(data_all.describe())
    print(data_all)
