# Solution guidance

Define Unicode normalization, token/span offsets, labels, and empty/long input.
Split by document, speaker, source, or time and remove near duplicates. Compare
lexical baselines; report class, length, language, and source slices. Persist
tokenizer, vocabulary, labels, model, and environment, then test offline reload,
privacy, and injection-like inputs.
