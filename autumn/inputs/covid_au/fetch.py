"""
This file imports Google mobility data and saves it to disk as a CSV.
"""
import os
import pandas as pd

from autumn import constants

# From covid19data.com.au GitHub https://github.com/M3IT/COVID-19_Data
DATA_URL = "https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_state_daily_change.csv"
COVID_AU_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "covid_au")
COVID_AU_CSV_PATH = os.path.join(COVID_AU_DIRPATH, "COVID_AU_state_daily_change.csv")


def fetch_covid_au_data():
    pd.read_csv(DATA_URL).to_csv(COVID_AU_CSV_PATH)