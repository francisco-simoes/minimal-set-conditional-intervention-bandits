def rowdf_to_dict(df):
    # Ensure the DataFrame has only one row
    if df.shape[0] != 1:
        raise ValueError("The DataFrame should have exactly one row.")

    # Convert the first row to a dictionary
    return df.iloc[0].to_dict()
