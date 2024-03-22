import pandas as pd
import streamlit as st

# Data
# Dataset url
WHR_URL = 'https://happiness-report.s3.amazonaws.com/2023/DataForTable2.1WHR2023.xls'

# Caching dataframe for perfs
@st.cache_data
def get_dataframe(url):
    df = pd.read_excel(url)
    # Prevent Streamlit from displaying year as float
    df['year'] = df['year'].astype(str)

    return df

whr = get_dataframe(WHR_URL)

# Structure
tabs = ["Data",
        "Visualization"]

tab = st.sidebar.radio("", options = tabs)

# Titles
st.title("World Happiness on Earth")
st.header(tab)

# Page 1 - Data
if tab == tabs[0]:
    # Full dataframe
    st.subheader("Show data")
    st.dataframe(whr)

    # Variables definitions
    if st.checkbox("Show variables definitions"):
        st.write("Variables")
        variables_definitions = {
            'Variable': [
                'Country name',
                'year',
                'Life Ladder',
                'Log GDP per capita',
                'Social support',
                'Healthy life expectancy at birth',
                'Freedom to make life choices',
                'Generosity',
                'Perceptions of corruption',
                'Positive affect',
                'Negative affect'
            ],
            'Role': [
                'Indexer',
                'Indexer',
                'Target',
                'Feature',
                'Feature',
                'Feature',
                'Feature',
                'Feature',
                'Feature',
                'Feature',
                'Feature'
            ],
            'Definition': [
                "Country described by the data of the record.",
                "Year of data recording for the row.",
                "Happiness level of a country according to the Cantril ladder, a scale between 0 and 10.",
                "Gross Domestic Product (GDP) per capita. This column provides information about the size and performance of the economy.",
                "Ratio of respondents who answered 'YES' to the question: 'If you encounter difficulties, do you have relatives or friends you can count on to help you?'",
                "Measures the physical and mental health of a country's population, based on data provided by the World Health Organization (WHO).",
                "Ratio of respondents who answered 'YES' to the question: 'Are you satisfied or dissatisfied with your freedom of choice/action?'",
                "Ratio of respondents who answered 'YES' to the question: 'Did you donate money to a charity last month?'",
                "Perception by the population of the level of corruption in their country (at both political - institutions - and economic - businesses - levels).",
                "Average of positive or negative responses given in relation to three emotions: laughter, pleasure, and interest.",
                "Average of positive or negative responses given in relation to three emotions: concern, sadness, and anger."
            ]
        }

        st.table(variables_definitions)

    # Variables dtypes and missing values
    st.subheader("Infos")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.dataframe(whr.dtypes, column_config = {
            "": "Column",
            "0": "Type"
        }, use_container_width = True)

    with info_col2:
        st.dataframe(whr.isna().sum(), column_config = {
            "": "Column",
            "0": "NaNs"
        }, use_container_width = True)

    # Stats
    st.subheader("Stats")
    st.dataframe(whr.describe().drop(index = ['count']), use_container_width = True)

    # Additional informations
    st.subheader("Misc")
    st.write("- Number of rows :", len(whr))
    st.write("- Number of countries:", whr['Country name'].nunique())
    st.write("- Years - from", int(whr['year'].min()), "to", int(whr['year'].max()))

if tab == tabs[1]:
    # History by country
    st.subheader("History by country")

    # Filter by country and variable
    hist_col1, hist_col2 = st.columns(2)

    with hist_col1:
        hist_country = st.selectbox(
            "Choose a country",
            sorted(whr['Country name'].unique())
        )

        hist_mean_by_year = st.checkbox("Show mean by year for all countries", value = True)

    with hist_col2:
        hist_variable = st.selectbox(
            "Choose a variable",
            whr.columns.drop(['Country name', 'year'])
        )

        hist_mean_by_country = st.checkbox("Show mean for this country accross all years")
    
    means_by_year = whr.drop(columns = ['Country name']).groupby('year').mean().reset_index()
    means_by_year.columns = ['year'] + list(map(lambda x: x + ' Mean by year', whr.columns.drop(['Country name', 'year'])))

    means_by_country = whr.drop(columns = ['year']).groupby('Country name').mean().reset_index()
    means_by_country.columns = ['Country name'] + list(map(lambda x: x + ' Mean by Country name', whr.columns.drop(['Country name', 'year'])))

    whr_with_means_by_year = pd.merge(whr,
                                      means_by_year,
                                      on = 'year',
                                      how = 'left')
    
    whr_with_means_by_year_and_country = pd.merge(whr_with_means_by_year,
                                                  means_by_country,
                                                  on = 'Country name',
                                                  how = 'left')
    
    hist_df = whr_with_means_by_year_and_country[whr_with_means_by_year_and_country['Country name'] == hist_country]

    hist_lines = [hist_variable]

    if hist_mean_by_year:
        hist_lines += [hist_variable + ' Mean by year']

    if hist_mean_by_country:
        hist_lines += [hist_variable + ' Mean by Country name']

    st.line_chart(hist_df, x = 'year', y = hist_lines)
