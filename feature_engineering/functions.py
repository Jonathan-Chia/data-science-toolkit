def list_categorical_columns(df):
    s = df.dtypes == "object"
    categorical_columns = list(s[s].index)
    return categorical_columns


def list_numerical_columns(df):
    s = (df.dtypes == "float64") | (df.dtypes == "int64")
    numerical_columns = list(s[s].index)
    return numerical_columns
