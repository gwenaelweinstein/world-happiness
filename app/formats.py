import dataframes as dfr

# Render style for variable
def var(var):
    return ":red[***" + dfr.get_label(var) + "***]"

# Render style for emphasis
def em(txt):
    return ":grey[***" + txt + "***]"

# Render style for quote
def cite(txt):
    return "> *" + txt + "*"
