# To generate all the figures and tables for the manuscript

## Preliminary steps
1. Put the PBI database containing all percentile outputs for all scenarios in "apps/tuberculosis/regions/marshall_islands/outputs/pbi_databases". 
2. Put a copy of the same PBI database in "data/outputs/calibrate/tuberculosis/marshall-islands/yyyy-mm-dd/" (to allow Step 3)
3. Use Streamlit to create a raw csv file containing the posterior percentiles.
4. Put the csv file into "apps/tuberculosis/regions/marshall_islands/outputs/parameter_posteriors".


## Generating the output files
Run the following script: "apps/tuberculosis/regions/marshall_islands/outputs/main_script.py".

All the figures and tables will be available from "apps/tuberculosis/regions/marshall_islands/outputs/figures". 


# Scenario list
The scenarios indexing should follow the definitions presented in ""apps/tuberculosis/regions/marshall_islands/README.md".
