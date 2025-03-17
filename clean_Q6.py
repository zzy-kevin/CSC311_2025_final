import numpy as np
import csv
import pandas as pd


def replace_common_words_Q6(value):
    """
    Milk tea should be not include tea since "tea" is in "milk tea"
    Everything should be lower case!
    """
    value_lower = str(value).lower()
    categories = []

    # Check each condition in order and append if any keyword is present and not excluded by more specific categories

    # Condition for miso
    if any(word in value_lower for word in
           ["miso", "japan", "ramune", "sake", "soju", "matcha", "calpis", "soy sauce"]):
        categories.append("miso")

    # Condition for coke
    if any(word in value_lower for word in ["coke", "cola", "pepsi"]):
        categories.append("coke")

    # Condition for iced tea
    if any(word in value_lower for word in ["iced tea", "nestea", "lemon", "ice tea"]):
        categories.append("iced tea")

    # Condition for bubble tea
    if any(word in value_lower for word in ["milk tea", "bubble tea", "boba"]):
        categories.append("bubble tea")

    # Condition for soda
    if any(word in value_lower for word in ["sprite", "root beer", "soda", "ginger", "crush", "pop", "fanta", "7up",
                                            "fruitopia", "mountain dew", "soft", "pepper", "dry", "carbonated"]):
        categories.append("soda")

    # Condition for tea (only if not already covered by iced tea or bubble tea)
    if "tea" in value_lower and not any(cat in categories for cat in ["iced tea", "bubble tea"]):
        categories.append("tea")

    # Condition for soup
    if "soup" in value_lower:
        categories.append("soup")

    # Condition for wine
    if any(word in value_lower for word in ["wine", "beer", "alcohol", "cocktail", "champagne", "rum"]):
        categories.append("wine")

    # Condition for juice
    if any(word in value_lower for word in ["juice", "smoothie"]):
        categories.append("juice")

    # Condition for milk (only if not already covered by bubble tea)
    if "milk" in value_lower and "bubble tea" not in categories:
        categories.append("milk")

    # Condition for sparkling water
    if "sparkling water" in value_lower:
        categories.append("sparkling water")

    # Condition for water (only if not already covered by sparkling water)
    if "water" in value_lower and "sparkling water" not in categories:
        categories.append("water")

    # Condition for none
    if any(word in value_lower for word in ["no", "nan"]):
        categories.append("none")

    # Condition for ayran
    if "ayran" in value_lower:
        categories.append("ayran")

    # Condition for any
    if "any" in value_lower and categories == []:
        categories.append("any")

    # If no categories matched, append 'other'
    if not categories:
        categories.append("other")

    return ", ".join(categories)


def clean_Q6(df):
    """
    # This cleans Q6
    :param df: the input dataframe to be cleaned
    :return: A df with Q6 cleaned
    """
    df_clean_Q6 = df.copy()
    df_clean_Q6['Q6'] = df_clean_Q6['Q6'].apply(replace_common_words_Q6)
    return df_clean_Q6


if __name__ == '__main__':

    df = pd.read_csv("cleaned_data_combined_modified.csv")

    # renaming to shorten the name, are you guys ok with this naming convention?
    new_column_name = ["id", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "target"]
    df.columns = new_column_name

    # cleans and return df with cleaned Q6
    df_cleaned_Q6 = clean_Q6(df)
    print(df_cleaned_Q6["Q6"])

    # Prints the original drink that got relabelled to "other"
    other_data = df[df_cleaned_Q6['Q6'].str.startswith('other')]['Q6']
    print(", ".join(other_data.to_list()))

    # Prints the original drink that got relabelled to "other"
    other_data = df[df_cleaned_Q6['Q6'].str.startswith('other')]['Q6']
    print(", ".join(other_data.to_list()))

    print(df_cleaned_Q6["Q6"][647])
    print(df["Q6"][647])