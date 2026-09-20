# Data provenance

[Sales Transactions Weekly, UCI](https://archive.ics.uci.edu/dataset/396/sales+transactions+dataset+weekly), Tan, J. (2014), [DOI 10.24432/C5XS4Q](https://doi.org/10.24432/C5XS4Q). Dataset license: CC BY 4.0. Code license: MIT.

The original file contains 811 product rows and 52 weekly purchase-quantity columns, W0–W51: 42,172 product-week observations. No calendar dates are provided; week indices are retained rather than invented dates. SHA256: `6f939a928c04820bc291a6184f4f1d0be694b8de58d949e808852804f25969c3`.

Read only Product_Code and raw W0–W51 quantities. MIN, MAX and Normalized columns summarize the entire history and could reveal future values, so they are excluded. Zero recorded purchases remain zeros. There is no availability, stockout, price, cost or supplier-capacity information; recorded purchases are an observed-demand proxy. All inventory states, lead times, capacities and cost penalties are scenario assumptions.

Download with `python scripts/download_data.py`. Raw data are excluded from Git and included in the downloadable ZIP for offline use.

Chronology: observe W0–W23 for the initial fit; select boosting complexity at origins 24–27, whose three-week targets end by W29; refit after W29; calibrate at origins 30–36, whose targets end at W38; start decisions at W39. Fully observable three-week forecast evaluation uses origins 39–49. Inventory simulation runs W39–W51.

ABC classes and the 30-product operational cohort are determined using W0–W23 only. The cohort samples ten active products per class with seed 42, rather than using test-period volume to choose easy products. All 811 products enter the forecasting evaluation.
