import dataframes as dfr
import streamlit as st

# Render style for variable
def var(var):
    return ":red[***" + dfr.get_label(var) + "***]"

# Render style for emphasis
def em(txt):
    return ":grey[***" + txt + "***]"

# Render style for quote
def cite(txt):
    return "> *" + txt + "*"

# Render error message
def error(txt):
    st.error(txt, icon='❌')

# Render warning message
def warning(txt):
    st.warning(txt, icon='⚠️')
