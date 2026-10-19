# Data

All experiments use the **IoT Greenhouse Irrigation Sensor Dataset** (version 1):

> I. Keok, "IoT Greenhouse Irrigation Sensor Dataset," Mendeley Data, V1, 2026.
> doi: [10.17632/2tvpsht4s4.1](https://doi.org/10.17632/2tvpsht4s4.1)

The dataset is released under CC BY 4.0, so a copy of `greenhouse_dataset.csv` is kept in this folder with attribution.
If you prefer to fetch it yourself, download it from the Mendeley page and place it here as `data/greenhouse_dataset.csv`.

Quick integrity check of the file used in the paper:

| Property | Value |
|---|---|
| Rows (excluding header) | 95,669 |
| Separator | `;` |
| Columns | `Timestamp; Soil Percent; Soil Raw; Suhu Tanah; Suhu Udara; Humidity; Label` |
| Labelled rows | 53,209 (53,001 × 0, 208 × 1) |
| Unlabelled rows | 42,460 (not used) |

`Suhu Tanah` is soil temperature and `Suhu Udara` is air temperature (Indonesian).
The Mendeley description also mentions a confidence value produced by the logging system; that column is not present in this CSV and is never used as a predictor.
