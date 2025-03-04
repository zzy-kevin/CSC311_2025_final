import numpy as np
import csv
import pandas as pd

df = pd.read_csv("cleaned_data_combined_modified.csv")

print(df.head())
print(df.describe())

# renaming to shorten the name, are you guys ok with this naming convention?
new_column_name = ["id", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "target"]
df.columns = new_column_name


def replace_common_words_Q6(value):
    """
    Note that we should rank these if statement in order of importance since for multiple drink we only return one.
    For example miso comes before all else because if someone said miso its probably sushi so Q6 should be miso not coke etc.
    Also milk tea should be before tea since "tea" is in "milk tea"
    Everything should be lower case!
    """
    if "miso" in value.lower() or "japan" in value.lower() or "ramune" in value.lower() or "sake" in value.lower() \
            or "soju" in value.lower() or "matcha" in value.lower() or "calpis" in value.lower()\
            or "soy sauce" in value.lower():
        # this has all the typical sushi and Japan related drink
        return "miso"
    if "coke" in value.lower() or "cola" in value.lower() or "pepsi" in value.lower():
        return "coke"
    if "iced tea" in value.lower() or "nestea" in value.lower() or "lemon" in value.lower() or "ice tea" in value.lower():
        # also includes lemonade
        return "iced tea"
    if "milk tea" in value.lower() or "bubble tea" in value.lower() or "boba" in value.lower():
        return "bubble tea"
    if "sprite" in value.lower() or "root beer" in value.lower() or "soda" in value.lower() or "ginger" in value.lower() \
            or "crush" in value.lower() or "pop" in value.lower() or "fanta" in value.lower() or "7up" in value.lower() \
            or "fruitopia" in value.lower() or "mountain dew" in value.lower() or "soft" in value.lower() \
            or "pepper" in value.lower() or "fruitopia" in value.lower() or "dry" in value.lower() or "carbonated" in value.lower():
        # all carbonated drink other than coke and sparkling water
        return "soda"
    if "tea" in value.lower():
        return "tea"
    if "soup" in value.lower():
        return "soup"
    if "wine" in value.lower() or "beer" in value.lower() or "alcohol" in value.lower() or "cocktail" in value.lower()\
            or "champagne" in value.lower() or "rum" in value.lower():
        return "wine"
    if "juice" in value.lower() or "smoothie" in value.lower():
        return "juice"
    if "milk" in value.lower():
        return "milk"
    if "sparkling water" in value.lower():
        return "sparkling water"
    if "water" in value.lower():
        return "water"
    if "no" in value.lower() or "nan" in value.lower():
        # also includes none
        return "none"
    if "ayran" in value.lower():
        return "ayran"
    return "other"



def clean_Q6(df):
    """
    # This cleans Q6
    :param df: the input dataframe to be cleaned
    :return: A df with Q6 cleaned
    """
    df_clean_Q6 = df.copy()
    df_clean_Q6['Q6'] = df_clean_Q6['Q6'].apply(replace_common_words_Q6)
    return df_clean_Q6

# cleans and return df with cleaned Q6
df_cleaned_Q6 = clean_Q6(df)
print(df_cleaned_Q6["Q6"])

# Prints the original drink that got relabelled to "other"
other_data = df[df_cleaned_Q6['Q6'].str.startswith('other')]['Q6']
print(", ".join(other_data.to_list()))