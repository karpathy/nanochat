
---
12/12
* what does tokenizer consists? 

1. Vocabulary list (8192 tokens)  
2. Token → ID mapping  
3. ID → Token mapping  
4. BPE merge rules  
5. Special tokens  
6. Normalization rules  
7. Pre-tokenization rules  
8. Post-processing rules  
9. Decoder logic


2. train a tokenizer
-> get the frequency of each token please

3. basic metrics of a tokenizer 

4. grouping
Group by token frequency
Group by embedding similarity

backup:
Group by distribution across datasets
Group by positional statistics
Group by token length (in characters or bytes)
Group by morphological similarity
Group by embedding similarity
Group by BPE merge lineage
Group by character type
Group by perplexity contribution
