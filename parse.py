import pandas as pd
import numpy as np

import clean_Q6
import clean_q2
import clean_Q4

def normalize_columns(df, cols):
    """
    Normalize specified columns (0/1 values) by dividing each value in a row
    by the square root of the sum of the row's values.
    - Handles division by zero by treating zero sums as 1 (resulting in 0/1 = 0).
    """
    # Compute sum of specified columns for each row
    row_sums = df[cols].sum(axis=1)

    # Compute sqrt of sums, replacing 0 with 1 to avoid division by zero
    sqrt_sums = np.sqrt(row_sums).replace(0, 1)

    # Divide each column by the sqrt of the row sum
    df[cols] = df[cols].div(sqrt_sums, axis=0)

    return df

def column_to_one_hot(df, column_name):
    """
    Convert a column with comma-separated categories into one-hot encoded columns.
    Handles whitespace around categories (e.g., "milk tea, tea" becomes "milk tea,tea").
    """
    # Clean the column: split, strip whitespace, and rejoin to ensure no spaces around categories
    cleaned_series = df[column_name].str.split(',').apply(
        lambda x: [s.strip() for s in x] if isinstance(x, list) else []
    ).str.join(',')

    # Explode to compute value counts (for debugging/analysis)
    exploded = cleaned_series.str.split(',').explode()
    value_counts = exploded.value_counts()
    print("Value counts after cleaning:\n", value_counts)

    # Create one-hot encoded columns
    one_hot = cleaned_series.str.get_dummies(sep=',').add_prefix(column_name + "_")

    normalize_one_hot = normalize_columns(one_hot, one_hot.columns)

    # Return the combined DataFrame and drop the original column
    return pd.concat([df, normalize_one_hot], axis=1).drop(columns=column_name, axis=1)

def clean_Q8(df):
    # Q8 is hot sauce amount which will be converted to numbers.
    # should we start at 0? since None should be 0?
    mapping = {
        'None': 0,
        'A little (mild)': 1,
        'A moderate amount (medium)': 2,
        'A lot (hot)': 3,
        'I will have some of this food item with my hot sauce': 4
    }
    df['Q8'] = df['Q8'].map(mapping)
    return df


def Q4_column_fixing(df, column):
    """
    Processes a column to split numeric values into Q4_price and text into Q4_tag.
    - If the value is numeric, Q4_price = value, Q4_tag = "None".
    - If the value is text, Q4_price = 9.99, Q4_tag = text.
    """
    # Convert to string and strip whitespace
    str_series = df[column].astype(str).str.strip()

    # Attempt to convert to numeric (handles both integers and floats)
    numeric_series = pd.to_numeric(str_series, errors='coerce')

    # Create Q4_price: use numeric value if valid, else 11.4 which is the average
    df['Q4_price'] = numeric_series.fillna(11.4)

    # Create Q4_tag: use original text if not numeric, else "None"
    df['Q4_tag'] = np.where(numeric_series.isna(), str_series, "No tag")

    # Drop the original column and return
    return df.drop(columns=[column])




#df = pd.read_csv("cleaned_data_combined_modified.csv")[:-1]  # we drop the last empty row for training data
# renaming to shorten the name, are you guys ok with this naming convention?

def parse_main(df):

    new_column_name = ["id", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "target"]
    df.columns = new_column_name

    # clean the data
    df_cleaned_Q6 = clean_Q6.clean_Q6(df)
    df_final = column_to_one_hot(df_cleaned_Q6, "Q6")

    df_final = clean_Q4.clean_Q4(df_final)
    df_final = Q4_column_fixing(df_final, "Q4")
    df_final = column_to_one_hot(df_final, "Q4_tag")

    df_final = clean_q2.clean_Q2(df_final)


    df_final = column_to_one_hot(df_final, "Q3")
    df_final = column_to_one_hot(df_final, "Q7")
    df_final = clean_Q8(df_final)
    # dropping these for now
    df_final = df_final.drop(columns=["id","Q5"])

    print(df_final.columns)
    print(df_final.iloc[1])
    print(df_final.iloc[10])

    return df_final