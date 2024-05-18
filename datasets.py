# Final label and definition for every variables
variables = {
    'country': {
        'label': "Country",
        'definition': "Country described by the data of the record."
    },
    'year': {
        'label': "Year",
        'definition': "Country described by the data of the record."
    },
    'target': {
        'label': "Life ladder",
        'definition': "Happiness level of a country according to the Cantril ladder, a scale between 0 and 10."
    },
    'gdp': {
        'label': "Log GDP per capita",
        'definition': "Gross Domestic Product (GDP) per capita. This column provides information about the size and performance of the economy."
    },
    'support': {
        'label': "Social support",
        'definition': "Ratio of respondents who answered *\"YES\"* to the question: *\"If you encounter difficulties, do you have relatives or friends you can count on to help you?\"*"
    },
    'life': {
        'label': "Healthy life expectancy at birth",
        'definition': "Measures the physical and mental health of a country's population, based on data provided by the World Health Organization (WHO)."
    },
    'freedom': {
        'label': "Freedom to make life choices",
        'definition': "Ratio of respondents who answered *\"YES\"* to the question: *\"Are you satisfied or dissatisfied with your freedom of choice/action?\"*"
    },
    'generosity': {
        'label': "Generosity",
        'definition': "Ratio of respondents who answered *\"YES\"* to the question: *\"Did you donate money to a charity last month?\"*"
    },
    'corruption': {
        'label': "Perception of corruption",
        'definition': "Perception by the population of the level of corruption in their country (at both political - institutions - and economic - businesses - levels)."
    },
    'positivity': {
        'label': "Positive affect",
        'definition': "Average of positive or negative responses given in relation to three emotions: laughter, pleasure, and interest."
    },
    'negativity': {
        'label': "Negative affect",
        'definition': "Average of positive or negative responses given in relation to three emotions: concern, sadness, and anger."
    }
}

# Infos for main and most recent datasets
datasets = {
    '2023': {
        'url': 'https://happiness-report.s3.amazonaws.com/2023/DataForTable2.1WHR2023.xls',
        'variables': {
            'country': "Country name",
            'year': "year",
            'target': "Life Ladder",
            'gdp': "Log GDP per capita",
            'support': "Social support",
            'life': "Healthy life expectancy at birth",
            'freedom': "Freedom to make life choices",
            'generosity': "Generosity",
            'corruption': "Perceptions of corruption",
            'positivity': "Positive affect",
            'negativity': "Negative affect"
        }
    },
    '2024': {
        'url': 'https://happiness-report.s3.amazonaws.com/2024/DataForTable2.1.xls',
        'variables': {
            'country': "Country name",
            'year': "year",
            'target': "Life Ladder",
            'gdp': "Log GDP per capita",
            'support': "Social support",
            'life': "Healthy life expectancy at birth",
            'freedom': "Freedom to make life choices",
            'generosity': "Generosity",
            'corruption': "Perceptions of corruption",
            'positivity': "Positive affect",
            'negativity': "Negative affect"
        }
    }
}
