import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

WHR_URL = 'https://happiness-report.s3.amazonaws.com/2023/DataForTable2.1WHR2023.xls'
whr = pd.read_excel(WHR_URL)
whr['year'] = whr['year'].astype(str)

st.title("World Happiness on Earth")

pages = ["Home",
        "Introduction",
        "Data exploration"]

page = st.sidebar.radio("", options=pages)

if page != pages[0]:
    st.header(page)

if page == pages[0]:
    st.write("An analysis of well-being on Earth based on data collected by the [World Happiness Report](https://worldhappiness.report/), whose survey aims to estimate the level of happiness by country based on socio-economic measures around health, education, corruption, economy, life expectancy, etc.")

    st.write("This work has been conducted as part of the [Data Analyst training program at DataScientest](https://datascientest.com/formation-data-analyst) in partnership with [Mines ParisTech PSL](https://executive-education.minesparis.psl.eu/).")

    st.image('assets/whr-cover.jpg', use_column_width=True)
    st.caption('Credits: image by [Phạm Quốc Nguyên](https://pixabay.com/fr/users/sanshiro-5833092)')

if page == pages[1]:
    st.write("The **World Happiness Report** is a publication of the [Sustainable Development Solutions Network](https://www.unsdsn.org/) for the United Nations, and powered by [Gallup World Poll](https://www.gallup.com/178667/gallup-world-poll-work.aspx) data. Its goal is to give more importance to happiness and well-being as criteria for assessing government policies.")

    st.write("**Gallup World Poll** ratings are based on the *Cantril Ladder*, where respondents are asked to evaluate their own life by giving a grade between 0 (the worst life they can think of) and 10 (the best one). The **World Happiness Report** adds 6 socio-economic measures and 2 daily feelings.")

    st.write("A report is published every year since 2012, with data from 2005 to the previous year of the publication.")

    st.write("> *With this project, we want to present this data using interactive visualizations and determine combinations of factors that explain why some countries are better ranked than others.*")

    st.caption("The current project and app have been done with data from the [2023 report](https://worldhappiness.report/ed/2023/). We may be able to test our process with data from 2024 at the end of this work.")

if page == pages[2]:
    st.subheader("Show data")
    st.dataframe(whr)

    st.write('''
        - The dataset is structured around 2 categorical variables (objects): :red[*Country name*] and :red[*year*]. Each record corresponds to the result of the survey for a given year and country: we call them *indexers*.
        - The *target* is represented by the variable :red[*Life Ladder*] (float).
        - All other variables (floats) are *features*.
    ''')
    
    with st.expander("Show variables definitions"):
        st.write('''
            :red[*Country name*]  
            Country described by the data of the record.
        ''')

        st.write('''
            :red[*year*]  
            Year of data recording for the row.
        ''')

        st.write('''
            :red[*Life Ladder*]  
            Happiness level of a country according to the Cantril ladder, a scale between 0 and 10.
        ''')

        st.write('''
            :red[*Log GDP per capita*]  
            Gross Domestic Product (GDP) per capita. This column provides information about the size and performance of the economy.
        ''')

        st.write('''
            :red[*Social support*]  
            Ratio of respondents who answered *"YES"* to the question: *"If you encounter difficulties, do you have relatives or friends you can count on to help you?"*
        ''')

        st.write('''
            :red[*Healthy life expectancy at birth*]  
            Measures the physical and mental health of a country's population, based on data provided by the World Health Organization (WHO).
        ''')

        st.write('''
            :red[*Freedom to make life choices*]  
            Ratio of respondents who answered *"YES"* to the question: *"Are you satisfied or dissatisfied with your freedom of choice/action?"*
        ''')

        st.write('''
            :red[*Generosity*]  
            Ratio of respondents who answered *"YES"* to the question: *"Did you donate money to a charity last month?"*
        ''')

        st.write('''
            :red[*Perceptions of corruption*]  
            Perception by the population of the level of corruption in their country (at both political - institutions - and economic - businesses - levels).
        ''')

        st.write('''
            :red[*Positive affect*]  
            Average of positive or negative responses given in relation to three emotions: laughter, pleasure, and interest.
        ''')

        st.write('''
            :red[*Negative affect*]  
            Average of positive or negative responses given in relation to three emotions: concern, sadness, and anger.
        ''')

    st.caption("Note that the variable :red[*year*] has been cast to an object to prevent Streamlit from displaying it as a float with thousand separators, knowing that it is not used for time series in our context.")

    st.subheader("Statistics")
    st.dataframe(whr.describe().drop(index=['count']), use_container_width=True)

    st.subheader("Details")
    st.write('''
        - Number of rows:''', len(whr),
        '''
        - Number of countries:''', whr['Country name'].nunique(),
        '''
        - Years: from''', int(whr['year'].min()), '''to''', int(whr['year'].max()))

    with st.expander("Show records by year and country"):
        st.bar_chart(whr, x='year', y='Country name')
    
    st.subheader("Distributions")
    for col in whr.drop(columns=['Country name', 'year']).columns:
        fig = px.histogram(whr, col, marginal='box')
        fig.update_layout(margin_b=0)
        fig.update_layout(margin_t=0)
        fig.update_layout(margin_r=0)
        fig.update_layout(margin_l=0)
        st.plotly_chart(fig, use_container_width=True)
    
    st.write('''
        - The target variable :red[*Life Ladder*] resembles a normal distribution.
        - All features show some outliers, which are not anomalies. For example, :blue[Haiti] shows a very low :red[*Healthy life expectancy at birth*] due to a natural disaster, while :blue[Venezuela] shows a very sharp fall in its :red[*Log GDP per capita*] following a major crisis of its economy.
    ''')
