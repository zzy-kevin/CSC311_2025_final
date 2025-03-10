import pandas as pd
import re

# df = pd.read_csv("cleaned_data_combined_modified.csv")
# q4 = df['Q4: How much would you expect to pay for one serving of this food item?']


def clean_q4(value):
    value = str(value)

    #instantly identify the food item under those cases:
    if 'pizza' or 'slice' in value.lower():
        return 'pizza'
    elif 'wrap' or 'shawarma' in value.lower():
        return 'shawarma'
    elif 'sushi' or 'pricy' or 'expensive' in value.lower():
        return 'sushi'
    # Find all numbers, including floats (with optional decimals)
    numbers = re.findall(r'\d+\.\d+|\d+', value)
    
    # Convert to float
    numbers = [float(num) for num in numbers]
    
    # return the average of all the numbers found in the string (won't be perfoect but works well enough)
    if numbers:
        return sum(numbers) / len(numbers)  
    else:   #if there is no number in thet string, check whether the number is in the words
        def words_to_num(word):
            num_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
        }
            return num_words.get(word.lower(), None)

        def extract_numbers_from_string(text):
            words = re.findall(r'\b[a-zA-Z]+\b', text)  # Extract words from string
            numbers = [words_to_num(word) for word in words if words_to_num(word) is not None]
            return numbers
        
        numbers = extract_numbers_from_string(value)
        if numbers:   #after converting the words to numbers, calculate the average
            return sum(numbers) / len(numbers)
        else:   #if there is still no number, flag it out (will probably change to assigning a default value)
            return 'flag'


# df['Q4'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(clean_q4)

# df.to_csv("q4.csv", index=False)


    
