---
title: Foundation model inference prompting and adaptation
slug: foundation-model-systems
level: practitioner
stage: generative-ai
estimated_hours: 14
prerequisites:
  - nlp-language-systems
  - deep-learning-framework-parity
learning_objectives:
  - Explain token prediction decoding context and inference tradeoffs
  - Design structured prompts and validate model outputs
  - Compare prompting retrieval fine-tuning and compression approaches
formats:
  - lesson
  - exercise
compute: free-gpu
status: draft
last_verified: 2026-09-03
---

# Foundation-model inference, prompting, and adaptation

Autoregressive language models estimate the next token conditioned on context. Tokenization determines input units and cost; context length is finite; decoding converts probabilities into outputs. Greedy decoding is repeatable but narrow, sampling introduces diversity, and temperature rescales uncertainty. None makes unsupported text factual.

Treat prompts as versioned program inputs. Separate instructions, trusted context, untrusted data, examples, and required output schema. Validate structured outputs with a parser and semantic checks. A model’s self-reported confidence is not calibrated evidence.

Choose the least invasive adaptation that meets measured needs: prompting for behavior expressed in context, retrieval for changing or attributable knowledge, fine-tuning for learned behavior/style/format, and distillation or quantization for serving constraints. Evaluate a baseline before changing the model.

## Exercise

Create a fixed evaluation set containing normal, ambiguous, unanswerable, long, multilingual, and adversarial inputs. Compare two prompt versions with identical model settings. Record prompt/model versions, token use, latency, schema-valid rate, task score, unsupported-claim rate, and paired examples. Predeclare the decision rule.

## Completion criteria

- [ ] Inputs and outputs have explicit schemas and validation.
- [ ] Evaluation separates model, prompt, and decoding changes.
- [ ] Unsupported answers are not counted as useful merely because fluent.
- [ ] Privacy, licensing, cost, and retention assumptions are documented.
