import pandas as pd
import re

def extract_number(value):
    """Extracts numeric values from a given input and returns a clean number."""
    if pd.isna(value):
        return None
    
    value = str(value).lower().strip()
    
    # Handle simple numeric values
    match = re.search(r'\d+', value)
    if match:
        return int(match.group())
    
    # Handle ranges (e.g., "7-Jun")
    range_match = re.findall(r'\d+', value)
    if range_match and len(range_match) == 2:
        return (int(range_match[0]) + int(range_match[1])) // 2  # Average of range
    
    # Handle text-based responses by counting commas and conjunctions
    if ',' in value or ' and ' in value:
        return value.count(',') + value.count(' and ') + 1
    
    return None  # Return None if no match is found

# Load the dataset
file_path = "cleaned_data_combined.csv"
df = pd.read_csv(file_path)

# Apply the cleaning function to the second column
df.iloc[:, 2] = df.iloc[:, 2].apply(extract_number)

# Compute the average of non-None values
valid_values = [v for v in df.iloc[:, 2] if pd.notna(v)]
average_value = sum(valid_values) // len(valid_values) if valid_values else 0

df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: x if pd.notna(x) else average_value)

# Save the cleaned dataset
df.to_csv("cleaned_data_output_q2.csv", index=False)
print("Data cleaned and saved as cleaned_data_output.csv")
