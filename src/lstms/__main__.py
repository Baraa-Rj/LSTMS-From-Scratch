import numpy as np
from Tokenizer import Tokenizer
def main() -> None:
    file_path = 'input.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)
    print(f'Encoded Text Sample: {encoded[:100]}')
    decoded = tokenizer.decode(encoded[:100])
    print(f'Decoded Text Sample: {decoded}')
    
   
if __name__ == "__main__":
    main()
