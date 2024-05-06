import dataframes as dfr
import ml_context as mlc
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def prepare(df, features):
    country_label = dfr.get_label('country')
    year_label = dfr.get_label('year')
    target_label = dfr.get_label('target')
    
    # Create test set with last year for each country
    df_last_years = df.groupby(country_label)[year_label].max()
    df_test = df[df.apply(lambda x: x[year_label] == df_last_years[x[country_label]], axis=1)]
    df_train = df.drop(df_test.index)

    # Separate target
    X_train = df_train[features]
    X_test = df_test[features]
    y_train = df_train[target_label]
    y_test = df_test[target_label]

    # Standardization
    sc = StandardScaler()
    X_train.loc[:, features] = sc.fit_transform(X_train[features])
    X_test.loc[:, features] = sc.transform(X_test[features])

    return X_train, X_test, y_train, y_test

@st.cache_resource
def execute(model, data, gridsearch=False):
    X_train, X_test, y_train, y_test = data

    # Get model
    match model:
        case "Linear Regression":
            trigger = LinearRegression()
        case "Decision Tree":
            trigger = DecisionTreeRegressor()
        case "Random Forest":
            trigger = RandomForestRegressor()
        case "Support Vector Regression (SVR)":
            trigger = SVR()
        case "K-Nearest Neighbors (KNN)":
            trigger = KNeighborsRegressor()

    # Optimize model
    if gridsearch:
        if mlc.models[model]['gridsearch']:
            gs = GridSearchCV(
                estimator=trigger,
                param_grid=mlc.models[model]['gridsearch'],
                scoring='r2'
            )

            gs.fit(X_train, y_train)

            return gs.best_estimator_

        else:
            return trigger.fit(X_train, y_train)
    
    else:
        return trigger.fit(X_train, y_train)

def metrics(model, data):
    X_train, X_test, y_train, y_test = data
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Coefficient of determination
    R2_train = model.score(X_train, y_train)
    R2_test = model.score(X_test, y_test)

    # Mean absolute error
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test = mean_absolute_error(y_test, y_pred_test)

    # Root mean squared error
    RMSE_train = mean_squared_error(y_train, y_pred_train, squared = False)
    RMSE_test = mean_squared_error(y_test, y_pred_test, squared = False)

    return (R2_train, R2_test), (MAE_train, MAE_test), (RMSE_train, RMSE_test)
