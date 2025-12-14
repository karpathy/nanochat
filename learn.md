
---
12/13
1. how does the training side know the vocab list len? 

base train, how to we make sure our dataset is enough?
how many tokens used in the training, how to make sure this one?
for one step, how many tokens are used for training?

2. train a model that 12 layers?
could we just pretrain and then try it?

3. how to modify the code here to let us train in block diffusion style? This one is important

4. how to set our tokens based on embedding?

5. encoder or decoder problem

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
Need conside the special tokens

backup:
Group by distribution across datasets
Group by positional statistics
Group by token length (in characters or bytes)
Group by morphological similarity
Group by embedding similarity
Group by BPE merge lineage
Group by character type
Group by perplexity contribution
