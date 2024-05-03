import dataframes as dfr
import datasets as dst
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
    st.write("The **World Happiness Report** is a publication of the [Sustainable Development Solutions Network](https://www.unsdsn.org/) for the United Nations, and powered by [Gallup World Poll](https://www.gallup.com/178667/gallup-world-poll-work.aspx) data. Its goal is to give more importance to happiness and well-being as criteria for assessing government policies.")

    st.write("**Gallup World Poll** ratings are based on the *Cantril Ladder*, where respondents are asked to evaluate their own life by giving a grade between 0 (the worst life they can think of) and 10 (the best one). The **World Happiness Report** adds 6 socio-economic measures and 2 daily feelings.")

    st.write("A report is published every year since 2012, with data from 2005 to the previous year of the publication.")

    st.write("> *With this project, we want to present this data using interactive visualizations and determine combinations of factors that explain why some countries are better ranked than others.*")

    st.caption("The current project and app have been done with data from the [2023 report](https://worldhappiness.report/ed/2023/). We may be able to test our process with data from 2024 at the end of this work.")

if page == pages[1]:
    st.subheader("Show data")
    st.dataframe(whr)

    st.write(f'''
        - The dataset is structured around 2 categorical variables (objects): :red[*{country_label}*] and :red[*{year_label}*]. Each record corresponds to the result of the survey for a given year and country: we call them *indexers*.
        - The *target* is represented by the variable :red[*{target_label}*] (float).
        - All other variables (floats) are *features*.
    ''')
    
    with st.expander("Show variables definitions"):
        for key, variable in dst.variables.items():
            st.write(f'''
                :red[*{variable['label']}*]  
                {variable['definition']}
            ''')

    st.caption(f"Note that the variable :red[*{year_label}*] has been cast to an object to prevent Streamlit from displaying it as a float with thousand separators, knowing that it is not used for time series in our context.")

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
        - The *target* variable :red[*{target_label}*] resembles a normal distribution.
        - All *features* show some outliers, which are not anomalies. For example, :grey[Haiti] shows a very low :red[*{dfr.get_label('life')}*] due to a natural disaster, while :grey[Venezuela] shows a very sharp fall in its :red[*{dfr.get_label('gdp')}*] following a major crisis of its economy.
    ''')

    st.subheader("Missing values")
    st.write(f"We know for sure that *target* (:red[*{target_label}*]) and *indexers* (:red[*{country_label}*] and :red[*{year_label}*]) show no missing values.")

    nans_per_variable_show_len_toggle = st.toggle("Show missing values compared to total records")

    fig = px.bar(whr[features].isna().sum())
    
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
        whr[[target_label] + features].corr(),
        text_auto='.2f',
        range_color=(-1, 1),
        color_continuous_scale=['#2E9AFF', '#FFFFFF', '#FF4B4B']
    )
    fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig, use_container_width=True)

    st.write(f'''
        - :red[*{dfr.get_label('gdp')}*], :red[*{dfr.get_label('support')}*] and :red[*{dfr.get_label('life')}*] have a strong positive correlation with :red[*{target_label}*], suggesting that these factors may be essential to the well-being of individuals.
        - :red[*{dfr.get_label('freedom')}*] and :red[*{dfr.get_label('positivity')}*] also show a moderately positive correlation with the target variable.
        - :red[*{dfr.get_label('corruption')}*] is moderately negatively correlated with :red[*{target_label}*], indicating that a high level of corruption perceived can be detrimental to general well-being.
        - :red[*{dfr.get_label('generosity')}*] shows relatively weak correlation scores.
        - The strongest correlation observed concerns the relationship between :red[*{dfr.get_label('gdp')}*] and :red[*{dfr.get_label('life')}*].
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

    st.write("The map shows some clusters, especially with high happiness levels for :grey[the West] and lowest ones in :grey[Africa] and :grey[Middle East].")

    st.write("The year", 2005, "is an exception, with very few countries participating to the survey and very high levels for the *target* variable, meaning this year can't be globally compared to the others. However,", 2005, "values are consistent with next years values for these countries.")

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
                st.error("No record for " + country_viz_country + " in " + whr[year_label].max() + ".", icon='❌')
    
    else:
        st.error("Not enough records available.", icon='❌')

if page == pages[3]:
    st.subheader("Filtering")
    st.write(f"As seen previously, the dataset is specific as it is structured around two indexing variables: the :red[*{dfr.get_label('country')}*] and the :red[*{dfr.get_label('year')}*].")
    
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
        countries_to_keep = max_year_per_country[max_year_per_country['year'].isin(['2022', '2021'])]
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

    st.write("Linear interpolation is an efficient method for estimating missing values ​​based on adjacent data points. Regarding our dataset, it means we can fill data gaps based on observed trends for each country.")

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

    st.write(f"All features are numerical and at the same scale, except for :red[*{dfr.get_label('gdp')}*] and :red[*{dfr.get_label('life')}*].")

    st.write("We may need to use standardization before modeling.")

    if st.session_state.whr_pp is None:
        st.session_state.whr_pp = whr_pp

if page == pages[4]:
    st.subheader("Classification")

    st.write(f"The machine learning problem at hand involves predicting a continuous variable, the :red[*{target_label}*], relying on 8 independent numerical variables as *features*.")

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
        st.error("You need to run *Preprocessing* step before processing modeling.", icon='❌')
    
    else:
        whr_pp = st.session_state.whr_pp

        whr_pp_last_years = whr_pp.groupby(country_label)[year_label].max()

        whr_pp_test = whr_pp[whr_pp.apply(lambda x: x[year_label] == whr_pp_last_years[x[country_label]], axis=1)]
        whr_pp_train = whr_pp.drop(whr_pp_test.index)

        modeling_features_options = st.multiselect(
            "Select features",
            features,
            features
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
            st.error("At least one feature is necessary.", icon='❌')
        
        else:
            X_train = whr_pp_train[modeling_features_options]
            X_test = whr_pp_test[modeling_features_options]
            y_train = whr_pp_train[target_label]
            y_test = whr_pp_test[target_label]

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

            st.write("> *No other model shows a better compromise between performance and robustness than :grey[Linear Regression] after :grey[Grid Search] optimization: either the model suffers from a great loss of performance, or we are not able to reduce overfitting enough.*")
