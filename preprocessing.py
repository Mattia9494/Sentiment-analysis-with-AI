# coding=utf-8
import re

# helper function to clean data using regular expressions
def process(data):
    # Remove HTML special entities (e.g. &amp;)
    data = re.sub(r'\&\w*;', '', data)
    #Convert @username
    data = re.sub(r'\@\w*', '', data)
    # Remove tickers
    data = re.sub(r'\$\w*', '', data)
    # To lowercase
    data = data.lower()
    # Remove hyperlinks
    data = re.sub(r'https?:\/\/.*\/\w*', '', data)
    # Remove hashtags
    data = re.sub(r'#\w*', '', data)
    # Remove words with 2 or fewer letters
    data = re.sub(r'\b\w{1,2}\b', '', data)
    # Remove whitespace (including new line characters)
    data = re.sub(r'\s\s+', ' ', data)
    # Remove single space remaining at the front of the data.
    data = data.lstrip(' ')
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    data = ''.join(c for c in data if c <= '\uFFFF')
    return data
