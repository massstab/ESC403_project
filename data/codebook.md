<h2>The code book</h2>
The sources of the raw data in the raw_data/ folder are:<br>
The main accident data: [RoadTrafficAccidentLocations](https://opendata.swiss/de/dataset/polizeilich-registrierte-verkehrsunfalle-auf-dem-stadtgebiet-zurich-seit-2011/resource/3bf3f12a-bf09-4e69-8cde-0df9e268d54b) <br>
Bycicle and pedestrian counting: [Daten der automatischen Fussgänger- und Velozählung - Viertelstundenwerte](https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo) <br>
Meteo data: [Stündlich aktualisierte Meteodaten](https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte) <br>
Traffic counting: [sid_dav_verkehrszaehlung_miv_OD2031_2020](https://data.stadt-zuerich.ch/dataset/sid_dav_verkehrszaehlung_miv_od2031/resource/44607195-a2ad-4f9b-b6f1-d26c003d85a2) <br>


In the `tidy_data/` folder there 5 different datasets. Including `data_merged.csv` which is the merged tidy dataset of
all other 4 tidy sets. For reproducibility reasons we provide also `tidyup.py` (esiting of raw data).

Because it was more convenient to have all the years in one
file, we merged the years 2011-2020 together with `raw_pretidy.py`.

In the `Descriptions/` folder there are also information about how the data was collected and other details to the raw data.

Here we describe the most important variables in the tidy files:
<h3>RoadTrafficAccidentLocations_cleaned.csv</h3>

<h4>AccidentType</h4>
0: Accident with skidding or self-accident<br>
1: Accident when overtaking or changing lanes<br>
2: Accident with rear-end collision<br>
3: Accident when turning left or right<br>
4: Accident when turning-into main road<br>
5: Accident when crossing the lane(s)<br>
6: Accident with head-on collision<br>
7: Accident when parking<br>
8: Accident involving pedestrian(s)<br>
9: Accident involving animal(s)<br>

<h4>AccidentSeverityCategory</h4>
1: Accident with fatalities<br>
2: Accident with severe injuries<br>
3: Accident with light injuries<br>
4: Accident with property damage<br>

<h4>RoadType</h4>
0: Motorway<br>
1: Expressway<br>
2: Principal road<br>
3: Minor road<br>
4: Motorway side installation<br>
9: Other

<h4>AccidentLocation_CHLV95_E/N</h4>
[Schweizer Landeskoordinaten LV95](https://www.zh.ch/de/planen-bauen/geoinformation/geodaten/koordinatensystem.html) <br>
First digit stands for east(2) and north(1).<br>
Reference point for coordinates:<br>
E (Ost) 	2'683'256.46 m 	 <br>
N (Nord) 	1'248'117.48 m 

