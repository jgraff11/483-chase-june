# Detecting Significant Hail from Radar Imagery Using a Random Forest Classifier

June Graff, Chase Hunter

<br>

### Abstract

...

<br>

### I. Background

A recurring problem in operational meteorology is predicting hail size with only radar data. The National Weather Service issues severe thunderstorm warnings for storms with radar-indicated hail surpassing the severe threshold (1.0 in), but issues stronger tags (considerable; destructive) for larger, significant-severe hail. The “now-casted” hail size can be an important factor in warnings, helping residents understand the danger that approaches and allowing them to act more effectively. 

In lieu of reports from the public or trained weather spotters, meteorologists are left with radar data from which they can estimate hail size themselves, or alternatively, employ algorithms or techniques aid in their estimation (Donavon & Jungbluth, 2007). One commonly used algorithm using radar data is MESH (Maximum Estimated Size of Hail). MESH has room to improve, having been found to greatly overestimate larger hail sizes in even a high-resolution hail report dataset (Wilson et al., 2009).  

Besides operational meteorology, this also has implications in hail research. One of the authors has experienced firsthand how limited even the best large-scale “ground truth” hail report datasets can be in several contexts, from investigating hail size associated with left-moving supercells to working towards building a hail “risk surface”. Should a more accurate way to predict hail from radar be developed, one could in theory construct a dataset of large spatial and temporal scale that could stand in for observations (or supplement them) with less uncertainty than MESH. 

To improve on algorithms like MESH, we aim to use machine learning. This is a more flexible approach that can take into account more factors than just the maximum reflectivity value or environmental temperature profile.  

### II. Methods

Our source of hail reports is the Storm Prediction Center, which compiles reports of severe hail across the United States into a database dating back to 1955. This database includes the date, time, location, and magnitude (size) of each hail report, which is sufficient for us to gather radar data and compare it to the reported hail size. 

The first step is pre-processing. Due to an excess of data and considering time constraints imposed by radar availability, we have chosen to only use hail reports during the 10-year period of 2011-2020. To ensure reliable quality of radar imagery, we also need to find the NWS WSR-88D radar site nearest to each report as well as its distance, filtering out reports that are too far to have high-resolution imagery. 

Next, we must download radar data from each report. We think this will simply be the scan closest to the time the report was sent, taken from the radar nearest to the report. This will come from the NWS’ archived level II data, which we are still figuring out how exactly to download, but we do have a lead. 

With radar data, we will begin by clipping the scan to only preserve where DBZ ≥ 30. While higher thresholds like 45dBz are typically used to identify hail cores (Tang et al., 2014), this more conservative approach may allow factors like storm mode to be better represented. From there, we will calculate features for our ML model, such as maximum and minimum ZDR, average DBZ, and area of ≥30dBz, ≥50dBz, etc. 

We plan to use a KMeans clustering model to differentiate between the hail sizes suggested by our radar-derived features. Rather than predict exact hail size, we have divided hail size into four bins: 1-2 inches, 2-3 in, 3-4 in, and 4+ in. This binned approach is more natural for KMeans, which would be able to find clusters for each hail size bin. 

To find the best model possible, we will first separate training, validation, and final testing data from our dataset. We will separate these by year and compare them to the full dataset to ensure they are each representative, if not, we will find another way to separate them so that they are. The training data will be used to train the KMeans model, the validation data to make initial predictions and tweak the model, and the testing data to evaluate the finished model’s performance. 

### III. Results

We pivoted from the KMeans clustering model to instead use a random forest, which we found was more suited to the task. Due to the sporadic nature of public reports as well as some sources of error that will be mentioned later, no clear clusters could be formed between any two features we developed. 

One (partially) unforeseen obstacle was the unreliability of radar data. While some files indeed lacked the ZDR and CC products we were looking for, removing those was trivial. The greater issue was that some scans were, in the area around our report, completely empty, partially empty (with an unnatural straight line cutting everything off suggesting some sort of corrupted or malformed data), or clearly representing a delayed report (no appreciable reflectivity returns near the report, which was likely sent after the storms moved out or dissipated). To mitigate this, we ended up having to manually verify several hundred reflectivity images. This took a lot of time and bottlenecked our process, leaving us with a decreased (but verified) sample. 

The random forest model was trained using only data from 2011-2016, validated and adjusted with data from 2017-2018, then given its final test with data from 2019-2020. 

Another difficulty we did not anticipate was the model simply performing very poorly with the original four hail size bins. Due to the small sample, we simply did not have enough representative cases for the larger bins, leading to hail size often being overestimated. Early iterations of the model received F-scores near 0.45, which is inadequate. To compensate, we adjusted our goalposts to instead only worry about sig-severe versus non-sig-severe hail (above or below the threshold of 2 inches). 

After reviewing expectations and updating our model, we received an initial F-score of 0.68 for our validation data. Looping through about seventy different configurations and picking the most skillful one, our final random forest model has an F-score of 0.70. This is far from ideal but shows great improvement from previous iterations. 

While the exact criteria are different, we do see a similar bias to MESH, in which we observe more erroneous predictions of large hail than we do of small hail (Wilson et al., 2009).  

It is unclear how much this can be improved given the limitations of public hail reports themselves, but we do believe we have identified some sources of error that could be improved in the future. As alluded to previously, a single randomly-selected report does not consistently represent a storm. It could be the peak hail size, some of the smallest hail, or somewhere in-between. A future step we did not previously consider would be to verify that only the largest report from each individual storm would be used, i.e. removing all but the largest report that took place within a certain lat/lon and time. This would prevent some inconsistency in how well the randomly sampled report actually represents its storm. 

Furthermore, our ZDR-derived parameters showed very little ability to discern between hail sizes. Despite filtering out typically noisy values (>5 and <-2), the maximum and minimum ZDR tends to be almost identical for each storm (right up against either limit) even after masking to DBZ ≥ 30. This is likely due to the influence of noisy values even within the area influenced by convective precipitation. It is not clear how to fix this exactly, but we could in theory look at ZDR minima sustained over an area, eliminating the influence that single noisy pixels can have on overall max/mins.

Finally, this sample size is smaller than ideal for the task at hand. Due to the slow manual verification process and many radar files not having the products we wanted, our final sample consisted of 383 reports overall. Dedicating more time to verifying more radar imagery or building an automatic or hybrid verification system could prove fruitful. 

<br>

### Availability Statement

The data needed for this project can be freely downloaded from <a href="https://niuits-my.sharepoint.com/:f:/g/personal/z2046057_students_niu_edu/IgARaqWHMUw7QpUnIpZPGc8KARJWsmG79ckiXBSJVaEPzP0?e=uYpbig">this public OneDrive</a>. `hail_v4.csv` is the final version of the file we used to train our model, and all data in it have been manually verified. To reproduce our process, you may download `1955-2024_hail.csv`, which is identical to the file found on <a href="https://www.spc.noaa.gov/wcm/#data">SPC's severe weather database page</a>.

### Acknowledgements

The authors would like to thank Dr. Haberlie for his guidance and patience during this project, and Brandon Weart for his contributions to the radar-downloading script.

### IV. Works Cited

Cintineo, J. L., T. M. Smith, V. Lakshmanan, H. E. Brooks, and K. L. Ortega, 2012: An Objective High-Resolution Hail Climatology of the Contiguous United States. Wea. Forecasting, 27, 1235–1248, https://doi.org/10.1175/WAF-D-11-00151.1. 

Donavon, R. A., and K. A. Jungbluth, 2007: Evaluation of a Technique for Radar Identification of Large Hail across the Upper Midwest and Central Plains of the United States. Wea. Forecasting, 22, 244–254, https://doi.org/10.1175/WAF1008.1. 

Murillo, E. M., C. R. Homeyer, and J. T. Allen, 2021: A 23-Year Severe Hail Climatology Using GridRad MESH Observations. Mon. Wea. Rev., 149, 945–958, https://doi.org/10.1175/MWR-D-20-0178.1. 

Ortega, K. L., J. M. Krause, and A. V. Ryzhkov, 2016: Polarimetric Radar Characteristics of Melting Hail. Part III: Validation of the Algorithm for Hail Size Discrimination. J. Appl. Meteor. Climatol., 55, 829–848, https://doi.org/10.1175/JAMC-D-15-0203.1. 

Tang, L., J. Zhang, C. Langston, J. Krause, K. Howard, and V. Lakshmanan, 2014: A Physically Based Precipitation–Nonprecipitation Radar Echo Classifier Using Polarimetric and Environmental Data in a Real-Time National System. Wea. Forecasting, 29, 1106–1119, https://doi.org/10.1175/WAF-D-13-00072.1.  

Wilson, C. J., Ortega, K., Lakshmanan, V., 2009: Evaluating Multi-Radar, Multi-Sensor Hail Diagnosis with High Resolution Hail Reports. https://caps.ou.edu/reu/reu08/finalpapers/Wilson-finalpaper.pdf 
