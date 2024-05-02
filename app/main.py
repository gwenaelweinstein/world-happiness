import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

WHR_URL = 'https://happiness-report.s3.amazonaws.com/2023/DataForTable2.1WHR2023.xls'

@st.cache_data
def main_df():
    df = pd.read_excel(WHR_URL)
    df['year'] = df['year'].astype(str)

    return df

whr = main_df()

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
    st.write("The **World Happiness Report** is a publication of the [Sustainable Development Solutions Network](https://www.unsdsn.org/) for the United Nations, and powered by [Gallup World Poll](https://www.gallup.com/178667/gallup-world-poll-work.aspx) data. Its goal is to give more importance to happiness and well-being as criteria for assessing government policies.")

    st.write("**Gallup World Poll** ratings are based on the *Cantril Ladder*, where respondents are asked to evaluate their own life by giving a grade between 0 (the worst life they can think of) and 10 (the best one). The **World Happiness Report** adds 6 socio-economic measures and 2 daily feelings.")

    st.write("A report is published every year since 2012, with data from 2005 to the previous year of the publication.")

    st.write("> *With this project, we want to present this data using interactive visualizations and determine combinations of factors that explain why some countries are better ranked than others.*")

    st.caption("The current project and app have been done with data from the [2023 report](https://worldhappiness.report/ed/2023/). We may be able to test our process with data from 2024 at the end of this work.")

if page == pages[1]:
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
        - Number of records:''', len(whr),
        '''
        - Number of countries:''', whr['Country name'].nunique(),
        '''
        - Years: from''', int(whr['year'].min()), '''to''', int(whr['year'].max()))

    with st.expander("Show records by year and country"):
        st.bar_chart(whr, x='year', y='Country name')

        st.caption("We observe that some countries show few records, and some - same or others - have not participated in the study recently. We may need to filter our dataset for modeling purposes.")
    
    st.subheader("Distributions")
    for col in whr.drop(columns=['Country name', 'year']).columns:
        fig = px.histogram(whr, col, marginal='box')
        fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.write('''
        - The *target* variable :red[*Life Ladder*] resembles a normal distribution.
        - All *features* show some outliers, which are not anomalies. For example, :grey[Haiti] shows a very low :red[*Healthy life expectancy at birth*] due to a natural disaster, while :grey[Venezuela] shows a very sharp fall in its :red[*Log GDP per capita*] following a major crisis of its economy.
    ''')

    st.subheader("Missing values")
    st.write("We know for sure that *target* (:red[*Life Ladder*]) and *indexers* (:red[*Country name*] and :red[*year*]) show no missing values.")

    nans_per_variable_show_len_toggle = st.toggle("Show missing values compared to total records")

    fig = px.bar(whr.drop(columns=['Life Ladder', 'Country name', 'year']).isna().sum())
    
    fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_layout(showlegend=False)
    fig.update_layout(yaxis_title="Number of missing values", xaxis_title=None)
    
    if nans_per_variable_show_len_toggle:
        fig.add_hline(y=len(whr), line_color='blue', line_width=1, annotation_text="Total records")
    
    st.plotly_chart(fig, use_container_width=True)

    st.write('''
        - All *features* show some missing values.
        - The dataset show''', whr.isna().sum().sum(), '''total missing values.
        - Missing values represent''', round(whr.isna().sum().sum() * 100 / (len(whr) * 8), 2), '''percent of all data in *features*.
    ''')

    st.subheader("Correlations")
    fig = px.imshow(
        whr.drop(columns=['Country name', 'year']).corr(),
        text_auto='.2f',
        range_color=(-1, 1),
        color_continuous_scale=['#2E9AFF', '#FFFFFF', '#FF4B4B']
    )
    fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig, use_container_width=True)

    st.write('''
        - :red[*Log GDP per capita*], :red[*Social support*] and :red[*Healthy life expectancy at birth*] have a strong positive correlation with :red[*Life Ladder*], suggesting that these factors may be essential to the well-being of individuals.
        - :red[*Freedom to make life choices*] and :red[*Positive affect*] also show a moderately positive correlation with the target variable.
        - :red[*Perceptions of corruption*] is moderately negatively correlated with :red[*Life Ladder*], indicating that a high level of corruption perceived can be detrimental to general well-being.
        - :red[*Generosity*] shows relatively weak correlation scores.
        - The strongest correlation observed concerns the relationship between :red[*Log GDP per capita*] and :red[*Healthy life expectancy at birth*].
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
        geo_target_year = st.select_slider("Select year", sorted(whr['year'].unique()), value=whr['year'].max())
    
    fig = px.scatter_geo(
        whr[whr['year'] == geo_target_year],
        scope=geo_target_scope,
        projection='natural earth',
        locations='Country name',
        locationmode='country names',
        color='Life Ladder',
        size='Life Ladder',
        color_continuous_scale=['#FF4B4B', '#FFFF00', '#09AB3B']
    )

    fig.update_layout(margin={'t': 0, 'b': 0, 'l': 0, 'r': 0})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_geos(bgcolor='rgba(0, 0, 0, 0)')

    st.plotly_chart(fig, use_container_width=True)

    st.write("The map shows some clusters, especially with high happiness levels for :grey[the West] and lowest ones in :grey[Africa] and :grey[Middle East].")

    st.write("The year", 2005, "is an exception, with very few countries participating to the survey and very high levels for the *target* variable, meaning this year can't be globally compared to the others. However,", 2005, "values are consistent with next years values for these countries.")

    st.subheader("Per country")
    country_viz_filter_col1, country_viz_filter_col2 = st.columns(2, gap='large')

    with country_viz_filter_col1:
        country_viz_country = st.selectbox(
            "Choose a country",
            sorted(whr['Country name'].unique())
        )
    
    with country_viz_filter_col2:
        country_viz_variable = st.selectbox(
            "Choose a variable",
            whr.columns.drop(['Country name', 'year'])
        )

    if len(whr[whr['Country name'] == country_viz_country][country_viz_variable].dropna()) > 1:
        fig = px.line(
            whr[whr['Country name'] == country_viz_country],
            x='year',
            y=country_viz_variable,
            markers=True,
            labels={"year": "", "value": country_viz_variable})
   
        fig.add_hline(
            y=whr[whr['Country name'] == country_viz_country][country_viz_variable].mean(),
            line_color='blue', line_width=1, annotation_text="Mean accross years"
        )
    
        fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        country_viz_metric_col1, country_viz_metric_col2, country_viz_metric_col3 = st.columns(3, gap='large')

        with country_viz_metric_col1:
            country_viz_metric1_list = whr[whr['Country name'] == country_viz_country][country_viz_variable].dropna().tolist()

            st.metric("Last record",
                round(country_viz_metric1_list[-1], 2),
                delta=round(country_viz_metric1_list[-1] - country_viz_metric1_list[-2], 2),
                help="compared to previous record"
            )

        with country_viz_metric_col2:
            country_viz_metric2 = whr[whr['Country name'] == country_viz_country][country_viz_variable].mean()
            country_viz_metric2_delta = whr[whr['Country name'] != country_viz_country][country_viz_variable].mean()

            st.metric("Mean",
                round(country_viz_metric2, 2),
                round(country_viz_metric2 - country_viz_metric2_delta, 2),
                help="compared to world mean"
            )

        with country_viz_metric_col3:
            if country_viz_country in whr[whr['year'] == whr['year'].max()]['Country name'].values:
                country_viz_metric3_sorted = whr[whr['year'] == whr['year'].max()].sort_values(country_viz_variable, ascending=False).reset_index()

                country_viz_metric3 = country_viz_metric3_sorted[country_viz_metric3_sorted['Country name'] == country_viz_country].index[0] + 1

                st.metric("Rank for " + whr['year'].max(),
                    country_viz_metric3,
                    help="out of " + str(len(country_viz_metric3_sorted)) + " countries"
                )
            
            else:
                st.error("No record for " + country_viz_country + " in " + whr['year'].max() + ".")
    
    else:
        st.error("Not enough records available.")

if page == pages[3]:
    st.subheader("Filtering")
    st.write("As seen previously, the dataset is specific as it is structured around two indexing variables: the :red[*Country name*] and the :red[*year*].")
    
    st.write("Consequently, the holdout part of the upcoming modeling process will not be conducted randomly: the test subset will consist of the most recent records.")
    
    st.write("We need to filter the dataset to keep a sufficiently representative amount of countries with records in a very restricted range of most recent year.")

    max_year_per_country = whr.groupby('Country name').agg({'year': 'max'})

    st.dataframe(
        max_year_per_country.value_counts().sort_index(ascending=False).head(),
        column_config={'count': st.column_config.Column("Number of records", width='medium')}
    )

    st.write("We can keep countries with records for", 2022, "and", 2021, ":")

    countries_to_keep = max_year_per_country[max_year_per_country['year'].isin(['2022', '2021'])]
    whr_pp = whr[whr['Country name'].isin(countries_to_keep.index)].reset_index(drop=True)

    st.code(
        '''
        # original dataframe is named 'whr'
        max_year_per_country = whr.groupby('Country name').agg({'year': 'max'})
        countries_to_keep = max_year_per_country[max_year_per_country['year'].isin(['2022', '2021'])]
        # new dataframe for preprocessing
        whr_pp = whr[whr['Country name'].isin(countries_to_keep.index)].reset_index(drop=True)
        ''',
        language='python'
    )

    st.write(
        "We kept", len(whr_pp), "records for", len(countries_to_keep), "countries,",
        "compared to", len(whr), "records for", whr['Country name'].nunique(), "countries before filtering."
    )

    st.subheader("Handling missing values")

    st.write("Linear interpolation is an efficient method for estimating missing values ​​based on adjacent data points. Regarding our dataset, it means we can fill data gaps based on observed trends for each country.")

    st.write("In return, this method requires handling missing values before the holdout step.")

    st.write("However, we observe very low variance and standard deviation in our dataset for each country, which significantly mitigates the risk of data leakage, as shown below:")

    var_countries = whr_pp.drop(columns = ['year', 'Life Ladder']).groupby('Country name').var()
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
    
    std_countries = whr_pp.drop(columns = ['year', 'Life Ladder']).groupby('Country name').std()
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
        for country in whr_pp['Country name'].unique():
            country_data = whr_pp[whr_pp['Country name'] == country]
            country_data = country_data.interpolate(method='linear', limit_direction='both')
            whr_pp.update(country_data)
        ''',
        language='python'
    )

    whrpp_nans_before = whr_pp.isna().sum().sum()

    for country in whr_pp['Country name'].unique():
        country_data = whr_pp[whr_pp['Country name'] == country]
        country_data = country_data.interpolate(method='linear', limit_direction='both')
        whr_pp.update(country_data)
    
    st.write("We reduced missing values from", whrpp_nans_before, "to", whr_pp.isna().sum().sum(), "corresponding to completely empty Series, that we can fill with mean values per year:")

    st.code(
        '''
        for col in whr_pp.drop(columns=['Country name', 'year']).columns:
            whr_pp[col] = whr_pp.groupby(['year'])[col].transform(lambda x: x.fillna(x.mean()))
        ''',
        language='python'
    )

    for col in whr_pp.drop(columns=['Country name', 'year']).columns:
        whr_pp[col] = whr_pp.groupby(['year'])[col].transform(lambda x: x.fillna(x.mean()))

    st.subheader("Feature scaling")

    st.write("All features are numerical and at the same scale, except for :red[*Log GDP per capita*] and :red[*Healthy life expectancy at birth*].")

    st.write("We may need to use standardization before modeling.")

    if st.session_state.whr_pp is None:
        st.session_state.whr_pp = whr_pp

if page == pages[4]:
    st.subheader("Classification")

    st.write("The machine learning problem at hand involves predicting a continuous variable, the :red[*Life Ladder*], relying on 8 independent numerical variables as *features*.")

    st.write("> *Therefore, this project pertains to an issue of linear regression.*")

    st.subheader("Performance metrics")
    
    st.markdown('''
        :grey[R\u00b2]  
        To estimate the contribution of each independent variable to explaining the value of the *target* variable, we must ensure that the model is able to explain a sufficient portion of the variance. We can do this with the coefficient of determination (R-Squared).
    ''')

    st.markdown('''
        :grey[MAE]  
        The coefficient of determination alone is not sufficient, as it does not take into account the magnitude of errors. Mean Absolute Error gives a first overview easy to interpret, with the advantage of being less sensitive to outliers.
    ''')

    st.markdown('''
        :grey[RMSE]  
        Mean Squared Error and Root Mean Squared Error are more sensitive to outliers than MAE, allowing to detect the potential presence of large discrepancies in predictions. RMSE is easier to interpret compared to the *target* variable and MAE, as it is on the same scale.
    ''')

    st.caption("If the accuracy of our predictions is a good performance indicator for our model, the focus will mainly be on interpreting these results.")

    st.subheader("Model selection")

    st.write("In order to determine the most suitable model for our problem, we compared the following models:")

    st.markdown('''
        :grey[Linear Regression]  
        Linear regression is relevant for our project, as it is simple to interpret and performs well on linear relationships between target and explanatory variables.
    ''')

    st.markdown('''
        :grey[Decision Tree]  
        This model is also quite straightforward to interpret. Moreover, the ability to visualize the path leading to the prediction can particularly highlight the role played by each explanatory variable.
    ''')

    st.markdown('''
        :grey[Random Forest]  
        Random forests are theoretically more generalizable than decision trees, and thus less prone to overfitting. With this model, we also seek to capture potential complex relationships between variables beyond what linear regression allows.
    ''')

    st.markdown('''
        :grey[Support Vector Regression (SVR)]  
        Similar to random forests, SVR is less prone to overfitting and can uncover complex, nonlinear relationships. It also performs well with small data volumes like ours.
    ''')

    st.markdown('''
        :grey[K-Nearest Neighbors (KNN)]  
        Similar to linear regression, this model is recognized for its simplicity and effectiveness. As the name suggests, it relies on the nearest samples to make predictions. Our dataset seems well suited for this model, as it contains country-level data over multiple years.
    ''')

    st.subheader("Process")

    if st.session_state.whr_pp is None:
        st.error("You need to run *Preprocessing* step before processing modeling.")
    
    else:
        whr_pp = st.session_state.whr_pp

        whr_pp_last_years = whr_pp.groupby("Country name")['year'].max()

        whr_pp_test = whr_pp[whr_pp.apply(lambda x: x['year'] == whr_pp_last_years[x['Country name']], axis=1)]
        whr_pp_train = whr_pp.drop(whr_pp_test.index)

        modeling_features_options = st.multiselect(
            "Select features",
            whr_pp.drop(columns=['Country name', 'year', 'Life Ladder']).columns.tolist(),
            whr_pp.drop(columns=['Country name', 'year', 'Life Ladder']).columns.tolist()
        )

        modeling_models = {
            'Linear Regression': (LinearRegression(), None),
            'Decision Tree': (DecisionTreeRegressor(), {
                'max_depth': [2, 4, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5],
                'max_features': [1.0, 'sqrt', 'log2']
            }),
            'Random Forest': (RandomForestRegressor(), {
                'n_estimators': [1, 10, 100],
                'max_depth': [2, 4, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5],
                'max_features': [1.0, 'sqrt', 'log2']
            }),
            'SVR': (SVR(), {
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.1, 0.01, 0.001]
            }),
            'KNN': (KNeighborsRegressor(), {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            })
        }

        modeling_model_select = st.selectbox(
            "Choose a model",
            modeling_models.keys()
        )

        if not modeling_features_options:
            st.error("At least one feature is necessary.")
        
        else:
            X_train = whr_pp_train[modeling_features_options]
            X_test = whr_pp_test[modeling_features_options]
            y_train = whr_pp_train['Life Ladder']
            y_test = whr_pp_test['Life Ladder']

            sc = StandardScaler()
            X_train.loc[:, modeling_features_options] = sc.fit_transform(X_train[modeling_features_options])
            X_test.loc[:, modeling_features_options] = sc.transform(X_test[modeling_features_options])

            modeling_model = modeling_models[modeling_model_select][0]

            modeling_model_fit = modeling_model.fit(X_train, y_train)

            y_pred_train = modeling_model_fit.predict(X_train)
            y_pred_test = modeling_model_fit.predict(X_test)

            R2_train = modeling_model_fit.score(X_train, y_train)
            R2_test = modeling_model_fit.score(X_test, y_test)

            MAE_train = mean_absolute_error(y_train, y_pred_train)
            MAE_test = mean_absolute_error(y_test, y_pred_test)

            RMSE_train = mean_squared_error(y_train, y_pred_train, squared = False)
            RMSE_test = mean_squared_error(y_test, y_pred_test, squared = False)

            modeling_metrics_metric_col1, modeling_metrics_metric_col2, modeling_metrics_metric_col3 = st.columns(3, gap='large')

            with modeling_metrics_metric_col1:
                st.metric("R2 Test",
                    round(R2_test, 2),
                    delta=round(R2_test - R2_train, 2),
                    help="R2 Train = " + str(round(R2_train, 2))
                )

            with modeling_metrics_metric_col2:
                st.metric("MAE Test",
                    round(MAE_test, 2),
                    delta=round(MAE_test - MAE_train, 2),
                    delta_color='inverse',
                    help="MAE Train = " + str(round(MAE_train, 2))
                )

            with modeling_metrics_metric_col3:
                st.metric("RMSE Test",
                    round(RMSE_test, 2),
                    delta=round(RMSE_test - RMSE_train, 2),
                    delta_color='inverse',
                    help="RMSE Train = " + str(round(RMSE_train, 2))
                )

            st.caption("Delta in these metrics show how Test results behave compared to Train results.")

            st.write('''
                - Every *feature* contributes to improvig the model's performance, even marginally: there is no way to enhance the model by removing any of them.
                - :grey[Linear regression] seems to have less favorable performance compared to other models, but it is also the most robust: all other models exhibit overfitting.
            ''')

            st.write("> *In order to find the best compromise between performance and robustness, we'll try to optimize each of these models with :grey[Grid Search] to determine the best hyperparameters.*")

            st.warning("Grid Search optimization with many models and parameters can take a very long time, proceed with caution.", icon='⚠️')

            gs_proceed = st.button("Proceed to Grid Search optimization", type='primary')
            
            if gs_proceed:
                gs_metrics = pd.DataFrame(columns=['Model', 'Set', 'R2', 'MAE', 'RMSE'])
                
                for gs_item in modeling_models:
                    if modeling_models[gs_item][1]:
                        param_grid = modeling_models[gs_item][1]
                        
                        gs = GridSearchCV(
                            estimator = modeling_models[gs_item][0],
                            param_grid = param_grid, cv = 5,
                            scoring = 'r2'
                        )
                        
                        gs.fit(X_train, y_train)

                        gs_model = gs.best_estimator_

                    else:
                        gs_model = modeling_models[gs_item][0].fit(X_train, y_train)
                    
                    gs_y_pred_train = gs_model.predict(X_train)
                    gs_y_pred_test = gs_model.predict(X_test)

                    gs_R2_train = gs_model.score(X_train, y_train)
                    gs_R2_test = gs_model.score(X_test, y_test)

                    gs_MAE_train = mean_absolute_error(y_train, gs_y_pred_train)
                    gs_MAE_test = mean_absolute_error(y_test, gs_y_pred_test)

                    gs_RMSE_train = mean_squared_error(y_train, gs_y_pred_train, squared = False)
                    gs_RMSE_test = mean_squared_error(y_test, gs_y_pred_test, squared = False)
                
                    gs_metrics.loc[len(gs_metrics.index)] = [
                        gs_item,
                        "Train",
                        gs_R2_train,
                        gs_MAE_train,
                        gs_RMSE_train
                    ]

                    gs_metrics.loc[len(gs_metrics.index)] = [
                        gs_item,
                        "Test",
                        gs_R2_test,
                        gs_MAE_test,
                        gs_RMSE_test
                    ]

                st.dataframe(gs_metrics, hide_index=True, use_container_width=True)

            st.write("> *No other model shows a better compromise between performance and robustness than :grey[Linear Regression] after Grid Search optimization: either the model suffers from a great loss of performance, or we are not able to reduce overfitting enough.*")