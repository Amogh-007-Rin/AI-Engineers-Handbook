# NLTK exercises

1. Extend the tokenizer contract to preserve accented Latin words. Add failing
   examples first and explain whether normalization uses NFC or NFKC.
2. Compare the offline regex tokenizer with `word_tokenize` on contractions,
   decimal values, email addresses, and emoji. Pin the required NLTK data asset.
3. Construct vocabulary on a training split, transform held-out documents, and
   report token coverage without updating the dictionary.
4. Build a unigram text classifier and compare it with majority and character
   n-gram baselines. Include errors caused by negation and casing.

Submit tests, a short decision log, and a table of at least eight boundary cases.
