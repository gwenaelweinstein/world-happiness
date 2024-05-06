import dataframes as dfr
import datasets as dst
import formats as fmt
import machine_learning as ml
import ml_context as mlc
import pandas as pd
import plotly.express as px
import streamlit as st

whr = dfr.get_df('2023')

country_label = dfr.get_label('country')
year_label = dfr.get_label('year')
target_label = dfr.get_label('target')
features = whr.drop(columns=[country_label, year_label, target_label]).columns.tolist()

if 'whr_pp' not in st.session_state:
    st.session_state.whr_pp = None

st.title("World Happiness on Earth")

pages = [
    "Introduction",
    "Data exploration",
    "Data visualization",
    "Preprocessing",
    "Modeling"
]

page = st.sidebar.radio("", options=pages)

if page != pages[0]:
    st.header(page)

if page == pages[0]:
    st.write("An analysis of well-being on Earth based on data collected by the [World Happiness Report](https://worldhappiness.report/), whose survey aims to estimate the level of happiness by country based on socio-economic measures around health, education, corruption, economy, life expectancy, etc.")

    st.write("This work has been conducted as part of the [Data Analyst training program at DataScientest](https://datascientest.com/formation-data-analyst) in partnership with [Mines ParisTech PSL](https://executive-education.minesparis.psl.eu/).")

    st.image('assets/whr-cover.jpg', use_column_width=True)
    st.caption('Credits: image by [Phạm Quốc Nguyên](https://pixabay.com/fr/users/sanshiro-5833092)')

    st.subheader("Context and goal")
    st.write(f"The {fmt.em("World Happiness Report")} is a publication of the [Sustainable Development Solutions Network](https://www.unsdsn.org/) for the United Nations, and powered by [Gallup World Poll](https://www.gallup.com/178667/gallup-world-poll-work.aspx) data. Its goal is to give more importance to happiness and well-being as criteria for assessing government policies.")

    st.write(f"{fmt.em("Gallup World Poll")} ratings are based on the {fmt.em("Cantril Ladder")}, where respondents are asked to evaluate their own life by giving a grade between 0 (the worst life they can think of) and 10 (the best one). The {fmt.em("World Happiness Report")} adds 6 socio-economic measures and 2 daily feelings.")

    st.write("A report is published every year since 2012, with data from 2005 to the previous year of the publication.")

    st.write(fmt.cite("With this project, we want to present this data using interactive visualizations and determine combinations of factors that explain why some countries are better ranked than others."))

    st.caption("The current project and app have been done with data from the [2023 report](https://worldhappiness.report/ed/2023/). We may be able to test our process with data from 2024 at the end of this work.")

if page == pages[1]:
    st.subheader("Show data")
    st.dataframe(whr)

    st.write(f'''
        - The dataset is structured around 2 categorical variables (objects): {fmt.var('country')}  and {fmt.var('year')}. Each record corresponds to the result of the survey for a given year and country: we call them {fmt.em("indexers")}.
        - The {fmt.em("target")} is represented by the variable {fmt.var('target')} (float).
        - All other variables (floats) are {fmt.em("features")}.
    ''')
    
    with st.expander("Show variables definitions"):
        for key, value in dst.variables.items():
            st.write(f'''
                {fmt.var(key)}  
                {value['definition']}
            ''')

    st.caption(f"Note that the variable {fmt.var('year')} has been cast to an object to prevent Streamlit from displaying it as a float with thousand separators, knowing that it is not used for time series in our context.")

    st.subheader("Statistics")
    st.dataframe(whr.describe().drop(index=['count']), use_container_width=True)

    st.subheader("Details")
    st.write('''
        - Number of records:''', len(whr),
        '''
        - Number of countries:''', whr[country_label].nunique(),
        '''
        - Years: from''', int(whr[year_label].min()), '''to''', int(whr[year_label].max()))

    with st.expander("Show records by year and country"):
        st.bar_chart(whr, x=year_label, y=country_label)

        st.caption("We observe that some countries show few records, and some - same or others - have not participated in the study recently. We may need to filter our dataset for modeling purposes.")

    st.subheader("Distributions")
    for col in [target_label] + features:
        fig = px.histogram(whr, col, marginal='box')
        fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.write(f'''
        - The {fmt.em("target")} variable {fmt.var('target')} resembles a normal distribution.
        - All {fmt.em("features")} show some outliers, which are not anomalies. For example, {fmt.em("Haiti")} shows a very low {fmt.var('life')} due to a natural disaster, while {fmt.em("Venezuela")} shows a very sharp fall in its {fmt.var('gdp')} following a major crisis of its economy.
    ''')

    st.subheader("Missing values")
    st.write(f"We know for sure that {fmt.em("target")} ({fmt.var('target')}) and {fmt.em("indexers")} ({fmt.var('country')} and {fmt.var('year')}) show no missing values.")

    nans_per_variable_show_len_toggle = st.toggle("Show missing values compared to total records")

    fig = px.bar(whr[features].isna().sum())
    
    fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_layout(showlegend=False)
    fig.update_layout(yaxis_title="Number of missing values", xaxis_title=None)
    
    if nans_per_variable_show_len_toggle:
        fig.add_hline(y=len(whr), line_color='blue', line_width=1, annotation_text="Total records")
    
    st.plotly_chart(fig, use_container_width=True)

    st.write(f'''
        - All {fmt.em("features")} show some missing values.
        - The dataset show''', whr.isna().sum().sum(), '''total missing values.
        - Missing values represent''', round(whr.isna().sum().sum() * 100 / (len(whr) * 8), 2), f'''percent of all data in {fmt.em("features")}.
    ''')

    st.subheader("Correlations")
    fig = px.imshow(
        whr[[target_label] + features].corr(),
        text_auto='.2f',
        range_color=(-1, 1),
        color_continuous_scale=['#2E9AFF', '#FFFFFF', '#FF4B4B']
    )
    fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig, use_container_width=True)

    st.write(f'''
        - {fmt.var('gdp')}, {fmt.var('support')} and {fmt.var('life')} have a strong positive correlation with {fmt.var('target')}, suggesting that these factors may be essential to the well-being of individuals.
        - {fmt.var('freedom')} and {fmt.var('positivity')} also show a moderately positive correlation with the target variable.
        - {fmt.var('corruption')} is moderately negatively correlated with {fmt.var('target')}, indicating that a high level of corruption perceived can be detrimental to general well-being.
        - {fmt.var('generosity')} shows relatively weak correlation scores.
        - The strongest correlation observed concerns the relationship between {fmt.var('gdp')} and {fmt.var('life')}.
    ''')

if page == pages[2]:
    st.subheader("Global preview")
    geo_target_col1, geo_target_col2 = st.columns(2, gap='large')
    
    with geo_target_col1:
        geo_target_scope = st.selectbox(
            "Zoom in",
            ('world', 'africa', 'asia', 'europe', 'north america', 'south america'),
            format_func=lambda x: x.title()
        )

    with geo_target_col2:
        geo_target_year = st.select_slider("Select year", sorted(whr[year_label].unique()), value=whr[year_label].max())
    
    fig = px.scatter_geo(
        whr[whr[year_label] == geo_target_year],
        scope=geo_target_scope,
        projection='natural earth',
        locations=country_label,
        locationmode='country names',
        color=target_label,
        size=target_label,
        color_continuous_scale=['#FF4B4B', '#FFFF00', '#09AB3B']
    )

    fig.update_layout(margin={'t': 0, 'b': 0, 'l': 0, 'r': 0})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_geos(bgcolor='rgba(0, 0, 0, 0)')

    st.plotly_chart(fig, use_container_width=True)

    st.write(f"The map shows some clusters, especially with high happiness levels for {fmt.em("the West")} and lowest ones in {fmt.em("Africa")} and {fmt.em("Middle East")}.")

    st.write("The year", 2005, f"is an exception, with very few countries participating to the survey and very high levels for the {fmt.em("target")} variable, meaning this year can't be globally compared to the others. However,", 2005, "values are consistent with next years values for these countries.")

    st.subheader("Per country")
    country_viz_filter_col1, country_viz_filter_col2 = st.columns(2, gap='large')

    with country_viz_filter_col1:
        country_viz_country = st.selectbox(
            "Choose a country",
            sorted(whr[country_label].unique())
        )
    
    with country_viz_filter_col2:
        country_viz_variable = st.selectbox(
            "Choose a variable",
            whr.columns.drop([country_label, year_label])
        )

    if len(whr[whr[country_label] == country_viz_country][country_viz_variable].dropna()) > 1:
        fig = px.line(
            whr[whr[country_label] == country_viz_country],
            x=year_label,
            y=country_viz_variable,
            markers=True,
            labels={year_label: "", "value": country_viz_variable})
   
        fig.add_hline(
            y=whr[whr[country_label] == country_viz_country][country_viz_variable].mean(),
            line_color='blue', line_width=1, annotation_text="Mean accross years"
        )
    
        fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        country_viz_metric_col1, country_viz_metric_col2, country_viz_metric_col3 = st.columns(3, gap='large')

        with country_viz_metric_col1:
            country_viz_metric1_list = whr[whr[country_label] == country_viz_country][country_viz_variable].dropna().tolist()

            st.metric("Last record",
                round(country_viz_metric1_list[-1], 2),
                delta=round(country_viz_metric1_list[-1] - country_viz_metric1_list[-2], 2),
                help="compared to previous record"
            )

        with country_viz_metric_col2:
            country_viz_metric2 = whr[whr[country_label] == country_viz_country][country_viz_variable].mean()
            country_viz_metric2_delta = whr[whr[country_label] != country_viz_country][country_viz_variable].mean()

            st.metric("Mean",
                round(country_viz_metric2, 2),
                round(country_viz_metric2 - country_viz_metric2_delta, 2),
                help="compared to world mean"
            )

        with country_viz_metric_col3:
            if country_viz_country in whr[whr[year_label] == whr[year_label].max()][country_label].values:
                country_viz_metric3_sorted = whr[whr[year_label] == whr[year_label].max()].sort_values(country_viz_variable, ascending=False).reset_index()

                country_viz_metric3 = country_viz_metric3_sorted[country_viz_metric3_sorted[country_label] == country_viz_country].index[0] + 1

                st.metric("Rank for " + whr[year_label].max(),
                    country_viz_metric3,
                    help="out of " + str(len(country_viz_metric3_sorted)) + " countries"
                )
            
            else:
                fmt.error("No record for " + country_viz_country + " in " + whr[year_label].max() + ".")
    
    else:
        fmt.error("Not enough records available.")

if page == pages[3]:
    st.subheader("Filtering")
    st.write(f"As seen previously, the dataset is specific as it is structured around two {fmt.em("indexing variables")}: the {fmt.var('country')} and the {fmt.var('year')}.")
    
    st.write("Consequently, the holdout part of the upcoming modeling process will not be conducted randomly: the test subset will consist of the most recent records.")
    
    st.write("We need to filter the dataset to keep a sufficiently representative amount of countries with records in a very restricted range of most recent year.")

    max_year_per_country = whr.groupby(country_label).agg({year_label: 'max'})

    st.dataframe(
        max_year_per_country.value_counts().sort_index(ascending=False).head(),
        column_config={'count': st.column_config.Column("Number of records", width='medium')}
    )

    st.write("We can keep countries with records for", 2022, "and", 2021, ":")

    countries_to_keep = max_year_per_country[max_year_per_country[year_label].isin(['2022', '2021'])]
    whr_pp = whr[whr[country_label].isin(countries_to_keep.index)].reset_index(drop=True)

    st.code(
        '''
        # original dataframe is named 'whr'
        max_year_per_country = whr.groupby('Country').agg({'Year': 'max'})
        countries_to_keep = max_year_per_country[max_year_per_country['Year'].isin(['2022', '2021'])]
        # new dataframe for preprocessing
        whr_pp = whr[whr['Country'].isin(countries_to_keep.index)].reset_index(drop=True)
        ''',
        language='python'
    )

    st.write(
        "We kept", len(whr_pp), "records for", len(countries_to_keep), "countries,",
        "compared to", len(whr), "records for", whr[country_label].nunique(), "countries before filtering."
    )

    st.subheader("Handling missing values")

    st.write(f"{fmt.em("Linear interpolation")} is an efficient method for estimating missing values based on adjacent data points. Regarding our dataset, it means we can fill data gaps based on observed trends for each country.")

    st.write("In return, this method requires handling missing values before the holdout step.")

    st.write("However, we observe very low variance and standard deviation in our dataset for each country, which significantly mitigates the risk of data leakage, as shown below:")

    var_countries = whr_pp.drop(columns=[year_label, target_label]).groupby(country_label).var()
    st.dataframe(
        var_countries.describe().loc[['mean', 'min', '25%', '50%', '75%', 'max']].round(2).transpose(),
        column_config={
            '': st.column_config.Column("Variance per country", width='medium'),
            '25%': st.column_config.Column("Q1"),
            '50%': st.column_config.Column("Q2"),
            '75%': st.column_config.Column("Q3")
        },
        use_container_width=True
    )
    
    std_countries = whr_pp.drop(columns=[year_label, target_label]).groupby(country_label).std()
    st.dataframe(
        std_countries.describe().loc[['mean', 'min', '25%', '50%', '75%', 'max']].round(2).transpose(),
        column_config={
            '': st.column_config.Column("Standard deviation per country", width='medium'),
            '25%': st.column_config.Column("Q1"),
            '50%': st.column_config.Column("Q2"),
            '75%': st.column_config.Column("Q3")
        },
        use_container_width=True
    )

    st.write("We can now interpolate missing values for each country, like this:")

    st.code(
        '''
        for country in whr_pp['Country'].unique():
            country_data = whr_pp[whr_pp['Country'] == country]
            country_data = country_data.interpolate(method='linear', limit_direction='both')
            whr_pp.update(country_data)
        ''',
        language='python'
    )

    whrpp_nans_before = whr_pp.isna().sum().sum()

    for country in whr_pp[country_label].unique():
        country_data = whr_pp[whr_pp[country_label] == country]
        country_data = country_data.interpolate(method='linear', limit_direction='both')
        whr_pp.update(country_data)
    
    st.write("We reduced missing values from", whrpp_nans_before, "to", whr_pp.isna().sum().sum(), "corresponding to completely empty Series, that we can fill with mean values per year:")

    st.code(
        '''
        for col in whr_pp.drop(columns=['Country', 'Year']).columns:
            whr_pp[col] = whr_pp.groupby(['Year'])[col].transform(lambda x: x.fillna(x.mean()))
        ''',
        language='python'
    )

    for col in whr_pp.drop(columns=[country_label, year_label]).columns:
        whr_pp[col] = whr_pp.groupby([year_label])[col].transform(lambda x: x.fillna(x.mean()))

    st.subheader("Feature scaling")

    st.write(f"All features are numerical and at the same scale, except for {fmt.var('gdp')} and {fmt.var('life')}.")

    st.write(fmt.cite("We may need to use standardization before modeling."))

    if st.session_state.whr_pp is None:
        st.session_state.whr_pp = whr_pp

if page == pages[4]:
    st.subheader("Classification")

    st.write(f"The machine learning problem at hand involves predicting a continuous variable, the {fmt.var('target')}, relying on 8 independent numerical variables as {fmt.em("features")}.")

    st.write(fmt.cite("Therefore, this project pertains to an issue of linear regression."))

    st.subheader("Performance metrics")

    for key, value in mlc.metrics.items():
        st.markdown(f'''
            {fmt.em(key)}  
            {value['definition']}
        ''')

    st.caption("If the accuracy of our predictions is a good performance indicator for our model, the focus will mainly be on interpreting these results.")

    st.subheader("Model selection")

    st.write("In order to determine the most suitable model for our problem, we compared the following models:")

    for key, value in mlc.models.items():
        st.markdown(f'''
            {fmt.em(key)}  
            {value['definition']}
        ''')

    st.subheader("Process")

    if st.session_state.whr_pp is None:
        fmt.error("You need to run *Preprocessing* step before processing modeling.")
    
    else:
        whr_pp = st.session_state.whr_pp

        modeling_features_options = st.multiselect(
            "Select features",
            features,
            features
        )

        modeling_model_select = st.selectbox(
            "Choose a model",
            mlc.models.keys()
        )

        if not modeling_features_options:
            fmt.error("At least one feature is necessary.")
        
        else:
            modeling_data = ml.prepare(whr_pp, modeling_features_options)

            modeling_model = ml.execute(modeling_model_select, modeling_data)

            modeling_model_metrics = ml.metrics(modeling_model, modeling_data)

            R2 = modeling_model_metrics[0]
            MAE = modeling_model_metrics[1]
            RMSE = modeling_model_metrics[2]

            modeling_metrics_metric_col1, modeling_metrics_metric_col2, modeling_metrics_metric_col3 = st.columns(3, gap='large')

            with modeling_metrics_metric_col1:
                st.metric("R\u00b2 Test",
                    round(R2[1], 2),
                    delta=round(R2[1] - R2[0], 2),
                    help="R\u00b2 Train = " + str(round(R2[0], 2))
                )

            with modeling_metrics_metric_col2:
                st.metric("MAE Test",
                    round(MAE[1], 2),
                    delta=round(MAE[1] - MAE[0], 2),
                    delta_color='inverse',
                    help="MAE Train = " + str(round(MAE[0], 2))
                )

            with modeling_metrics_metric_col3:
                st.metric("RMSE Test",
                    round(RMSE[1], 2),
                    delta=round(RMSE[1] - RMSE[0], 2),
                    delta_color='inverse',
                    help="RMSE Train = " + str(round(RMSE[0], 2))
                )

            st.caption("Delta in these metrics show how Test results behave compared to Train results.")

            st.write(f'''
                - Every {fmt.em("feature")} contributes to improvig the model's performance, even marginally: there is no way to enhance the model by removing any of them.
                - {fmt.em("Linear Regression")} seems to have less favorable performance compared to other models, but it is also the most robust: all other models exhibit overfitting.
            ''')

            st.write(fmt.cite(f"In order to find the best compromise between performance and robustness, we'll try to optimize each of these models with {fmt.em("Grid Search")} to determine the best hyperparameters."))

            fmt.warning("Grid Search optimization with many models and parameters can take a very long time, proceed with caution.")

            gs_proceed = st.button("Proceed to Grid Search optimization", type='primary')
            
            if gs_proceed:
                gs_metrics = pd.DataFrame(columns=['Model', 'Set', 'R2', 'MAE', 'RMSE'])
                
                for gs_item in mlc.models.keys():
                    gs_data = ml.prepare(whr_pp, features)

                    gs_model = ml.execute(gs_item, gs_data, gridsearch=True)

                    gs_model_metrics = ml.metrics(gs_model, gs_data)

                    gs_R2 = gs_model_metrics[0]
                    gs_MAE = gs_model_metrics[1]
                    gs_RMSE = gs_model_metrics[2]

                    gs_metrics.loc[len(gs_metrics.index)] = [
                        gs_item,
                        "Train",
                        gs_R2[0],
                        gs_MAE[0],
                        gs_RMSE[0]
                    ]

                    gs_metrics.loc[len(gs_metrics.index)] = [
                        gs_item,
                        "Test",
                        gs_R2[1],
                        gs_MAE[1],
                        gs_RMSE[1]
                    ]
                    
                st.dataframe(gs_metrics, hide_index=True, use_container_width=True)
            
            st.write(fmt.cite(f"No other model shows a better compromise between performance and robustness than {fmt.em("Linear Regression")} after {fmt.em("Grid Search")} optimization: either the model suffers from a great loss of performance, or we are not able to reduce overfitting enough."))
