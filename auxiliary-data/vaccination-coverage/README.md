# Vaccination Coverage

## Round 20

The [COVID_RD20_Vaccination_curves](./COVID_RD20_Vaccination_curves.csv) 
file contains vaccination coverage information for round 20.

Estimates have been extrapolated and smoothed to provide weekly data. 
The `"coverage"` column contains percent coverage data in percent, meaning 0.4 
represents 0.4 %, or in another example, 40 represents 40 %. 

For Scenario A:
- season 2025-2026: fall vaccine coverage operates from mid-August 2025 to mid-February 2026 

For Scenario B and D: 
- season 2025-2026: fall vaccine coverage operates from mid-August 2025 to mid-February 2026 
- season 2026-2027: same as 2025-2026 data, shifted a year. 

For Scenario C and E: 
- season 2025-2026: fall vaccine campaign operates from mid-August 2025 to mid-February 2026 
for all age groups. A second campaign for high-risk groups and 65+ starts mid-February 2026 
to mid-August 2026 corresponding to 0.5 \* coverage of the 2026-2027 campaign 
- season 2026-2027: same as 2025-2026 fall vaccination data, shifted by a year.

The data is available at state and national level, for multiple age groups 
(6mo-17, 18-49, 50-64, 65+) and risk group (high-risk, low-risk, overall)
with the associated population size (from 
[2024 Census](https://www.census.gov/data/datasets/time-series/demo/popest/2020s-state-detail.html) data). 
The data are expressed by epiweek (week starting Sunday) and scenario.

An illustration of the five scenarios vaccination curves for all age group at
national level is available in a [national_vaxcurve_r20](./national_vaxcurve_r20.png)
PNG file.

### Source

COVID-19 coverage: adults data are based on 2025-2026 
[National Immunization Survey Fall Respiratory Virus Module (NIS-FRVM): RespVaxView| Data | Centers for Disease Control and Prevention (cdc.gov)](https://data.cdc.gov/Vaccinations/National-Immunization-Survey-Fall-Respiratory-Viru/ee83-ukst/) 
and the children data are based on 
[Weekly Parental Intent for Vaccination and Cumulative Percentage of Children 6 Months -17 Years Who are Up to date with the COVID-19 Vaccines by Season, United States](https://data.cdc.gov/Child-Vaccinations/Weekly-Parental-Intent-for-Vaccination-and-Cumulat/ker6-gs6z/)

Flu coverage: adults data are based on flu coverage data from 2024-2025 season available in 
[National Immunization Survey Adult COVID Module (NIS-ACM): RespVaxView| Data | Centers for Disease Control and Prevention (cdc.gov)](https://data.cdc.gov/Vaccinations/National-Immunization-Survey-Adult-COVID-Module-NI/si7g-c2bs/about_data) 
and have been shifted by a year. Children data are based on 2025-2026 season 
[Weekly Cumulative Influenza Vaccination Coverage by Flu Season, Selected Demographics, and Race and Ethnicity Among Children 6 Months-17 Years, United States](https://data.cdc.gov/Child-Vaccinations/Weekly-Cumulative-Influenza-Vaccination-Coverage-b/vncy-2ds7/about_data) 


## Observed Data

The 
[Weekly Parental Intent for Vaccination and Cumulative Percentage of Children 6 Months -17 Years Who are Up to date with the COVID-19 Vaccines by Season](https://data.cdc.gov/Child-Vaccinations/Weekly-Parental-Intent-for-Vaccination-and-Cumulat/ker6-gs6z/about_data)
and the
[COVID-19 Vaccination Coverage, Overall and by Selected Demographics and Jurisdiction, Among Adults 18 Years and Older, by Season](https://data.cdc.gov/Vaccinations/COVID-19-Vaccination-Coverage-Overall-and-by-Selec/ksfb-ug5d/about_data)
is also downloaded weekly and stored in the [vax_cov](./vax_cov/) folder in CSV files 
with the download date added in the name of the files. Past files are stored in an 
[archive](./vax_cov/archive) folder.

> Weekly estimates of COVID-19 vaccination coverage and parental intent for vaccination among children 
through December 31, 2023, were calculated using data from the National Immunization Survey-Child COVID 
Module (NIS-CCM). (https://www.cdc.gov/nis/about/index.html).

> Weekly estimates of COVID-19 vaccination coverage and intent for vaccination among adults are 
calculated using data from the National Immunization Surveys (NIS)-Fall Respiratory Virus 
Module (NIS-FRVM) (https://www.cdc.gov/nis/about/index.html). 


## Archive 

### Round 19

The [COVID_RD19_Vaccination_curves](./COVID_RD19_Vaccination_curves.csv) file contains
vaccination coverage information for round 19.

Historical coverage is based on the 
[Adult National Immunization Survey]( https://data.cdc.gov/Vaccinations/National-Immunization-Survey-Adult-COVID-Module-NI/si7g-c2bs/about_data) and
[Children National Immunization Survey](https://data.cdc.gov/Child-Vaccinations/Weekly-Parental-Intent-for-Vaccination-and-Cumulat/ker6-gs6z/about_data)
for the period  Sep 2024 to March 2025. Estimates have been extrapolated 
and smoothed to provide  weekly data. This generates the data corresponding 
to the "Scenario == Historic coverage" observations. THe `"Cum.Coverage.Percent"`
column contains percent coverage data in percent, meaning `0.4` represents
`0.4 %`, or in another example, `40` represents `40 %`.

For Scenarios B & D, which represent the same timing of vaccination in 2025-26 
as in 2024-2025 we shift the historic coverage estimates by 365 days
For Scenarios C & F, which represent 1.5 month earlier timing of vaccination 
in 2025-26 as in 2024-2025, we shift the coverage estimates obtained for Scenarios 
B & D by 7 weeks earlier.

*Note* Scenario A data are not provided, it should be assumed that vaccine coverage 
is 0% in all age and risk groups in the 2025-26 season

The data is available at state and national level, for multiple age groups (6mo-17, 
18-49, 50-64, 65+) and risk group (high-risk, low-risk, overall). The data are
expressed by epiweek (week starting Sunday) and scenario.
