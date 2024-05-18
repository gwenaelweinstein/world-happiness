models = {
    "Linear Regression": {
        'definition': "Linear Regression is relevant for our project, as it is simple to interpret and performs well on linear relationships between target and explanatory variables.",
        'gridsearch': None
    },
    "Decision Tree": {
        'definition': "This model is also quite straightforward to interpret. Moreover, the ability to visualize the path leading to the prediction can particularly highlight the role played by each explanatory variable.",
        'gridsearch': {
            'max_depth': [2, 4, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5],
            'max_features': [1.0, 'sqrt', 'log2']
        }
    },
    "Random Forest": {
        'definition': "Random Forests are theoretically more generalizable than Decision Trees, and thus less prone to overfitting. With this model, we also seek to capture potential complex relationships between variables beyond what Linear Regression allows.",
        'gridsearch': {
            'n_estimators': [1, 10, 100],
            'max_depth': [2, 4, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5],
            'max_features': [1.0, 'sqrt', 'log2']
        }
    },
    "Support Vector Regression (SVR)": {
        'definition': "Similar to Random Forests, SVR is less prone to overfitting and can uncover complex, nonlinear relationships. It also performs well with small data volumes like ours.",
        'gridsearch': {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.1, 0.01, 0.001]
        }
    },
    "K-Nearest Neighbors (KNN)": {
        'definition': "Similar to Linear Regression, this model is recognized for its simplicity and effectiveness. As the name suggests, it relies on the nearest samples to make predictions. Our dataset seems well suited for this model, as it contains country-level data over multiple years.",
        'gridsearch': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}

metrics = {
    "R\u00b2": {
        'definition': "To estimate the contribution of each independent variable to explaining the value of the target variable, we must ensure that the model is able to explain a sufficient portion of the variance. We can do this with the coefficient of determination (R-Squared)."
    },
    "MAE": {
        'definition': "The coefficient of determination alone is not sufficient, as it does not take into account the magnitude of errors. Mean Absolute Error gives a first overview easy to interpret, with the advantage of being less sensitive to outliers."
    },
    "RMSE": {
        'definition': "Mean Squared Error and Root Mean Squared Error are more sensitive to outliers than MAE, allowing to detect the potential presence of large discrepancies in predictions. RMSE is easier to interpret compared to the target variable and MAE, as it is on the same scale."
    }
}
