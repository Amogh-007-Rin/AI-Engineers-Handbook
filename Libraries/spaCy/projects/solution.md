# Solution notes

`spacy.blank` prevents an implicit model download and makes the tokenizer part
of the tested artifact. Structured annotations are compared because equivalent
display text can conceal different offsets or labels. Production packages
should also pin the language model and check its metadata compatibility.
