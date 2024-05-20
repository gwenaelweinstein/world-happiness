import datasets as dst
import pandas as pd
import streamlit as st

# Get formatted dataframe
@st.cache_data
def get_df(year):
    # Create dataframe from url
    df = pd.read_excel(dst.datasets[year]['url'])

    # Rename columns
    rename_cols_dict = {dst.datasets[year]['variables'][key]: get_label(key) for key in dst.variables}
    df = df.rename(columns=rename_cols_dict)

    # Change year dtype for display purposes
    df[get_label('year')] = df[get_label('year')].astype(str)

    return df

# Get final label for a variable/column
@st.cache_data
def get_label(var):
    return dst.variables[var]['label']
