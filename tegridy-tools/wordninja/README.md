# wordninja
## Mirror copy of [wordninja](https://github.com/keredson/wordninja) by Derek Anderson

***

## Source code retrieved on 12/29/2024

***

### Installation

```sh
pip install wordninja
```

***

### Basic use example

```python
import re
import wordninja

def clean_and_separate_text(text):
    # Step 1: Clean text by removing unwanted symbols and numbers
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Step 2: Tokenize the cleaned text using wordninja
    tokens = wordninja.split(cleaned_text)
    
    # Step 3: Filter out empty strings and return the result
    separated_words = [token for token in tokens if token]
    
    return separated_words

# Example usage
text = "thisisahatandthisisa123test!withsymbols"
separated_words = clean_and_separate_text(text)
print(separated_words)
```

***

### Project Los Angeles
### Tegridy Code 2024
