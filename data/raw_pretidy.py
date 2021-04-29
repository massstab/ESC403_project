import os
import numpy as np
import pandas as pd

# This code is just used to combine the raw data from multiple files into one big file...

joined_csv = pd.read_csv('raw_data/pre_tidy_fussgaenger_velo/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv', index_col=None)
print(joined_csv)
print(joined_csv.head())
directory = 'raw_data/pre_tidy_fussgaenger_velo'
_, _, filenames = next(os.walk(directory))
for filename in filenames:
    if filename == '2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv':
        continue
    joined = os.path.join(directory, filename)
    print(filename)
    joined_csv = joined_csv.append(pd.read_csv(joined, ))
joined_csv.sort()
joined_csv.to_csv('raw_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv', index=False)
joined_csv.to_pickle('raw_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.pickle')
# joined_csv = pd.read_pickle('raw_data/Verkehrszaehlungen_werte_autos/sid_dav_verkehrszaehlung_miv_OD2031_2012-2020.pickle')
print(joined_csv)



