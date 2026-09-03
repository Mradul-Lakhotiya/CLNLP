# Experiment 05: Subword Tokenization and POS Tagging

**Course:** Computational Linguistics and Natural Language Processing (CLNLP) Lab  
**Experiment Name:** Experiment-5: Subword Tokenization and POS Tagging  
**Date:** 2026-09-03  

---

## 1. Aim and Objectives
- **Aim:** To implement advanced subword-level tokenization techniques using Byte Pair Encoding (BPE) and SentencePiece, and to apply Part-of-Speech (POS) tagging on contextual text using modern NLP libraries.
- **Objectives:**
  - [ ] Objective 1 (Exp 5.1): Implement subword tokenization using a BPE-based tokenizer, using both a pretrained model and training one from scratch.
  - [ ] Objective 2 (Exp 5.2): Implement subword tokenization using SentencePiece, using both a pretrained model and training one from scratch.
  - [ ] Objective 3 (Exp 5.3): Perform POS tagging on a sample sentence and display token, POS tag, and POS tag description using both spaCy and NLTK.
  - [ ] Objective 4 (Exp 5.4): Perform POS tagging on a large corpus using spaCy and calculate the frequency of each unique token along with its POS description.

---

## 2. Theoretical Background and Linguistic Concepts
- **Word vs Subword Tokenization:** Word-level tokenization is intuitive but suffers from the out-of-vocabulary (OOV) problem for rare words. Subword tokenization dynamically chunks words into recognizable sub-units (e.g., `playing` -> `play` + `ing`), significantly restricting the vocabulary size while gracefully handling unknown words.
- **Byte Pair Encoding (BPE):** An algorithm that begins with character-level tokens and iteratively merges the most frequently adjacent pairs. It inherently builds a dictionary of highly recurrent character strings.
- **SentencePiece:** An unsupervised text tokenizer developed by Google that operates directly on raw text sentences, treating spaces as a regular character (`_`). It avoids the need for a pre-tokenization step, making it highly effective for languages without explicit word boundaries.
- **Part-of-Speech (POS) Tagging:** The task of labeling each word in a sequence with its corresponding lexical category (Noun, Verb, Adjective, etc.) dependent on its syntactic context. POS tags illuminate the grammatical structure of a sentence.

---

## 3. Algorithm and Methodology
1. **Experiment 5.1 (BPE Tokenization):**
   - **Pretrained:** Use `transformers.AutoTokenizer` with a standard BPE model (e.g., `gpt2`).
   - **From Scratch:** Use `tokenizers.Tokenizer(BPE())`, train it on `data/input_text.txt`, and tokenize a sample sentence.
2. **Experiment 5.2 (SentencePiece Tokenization):**
   - **Pretrained:** Use `transformers.AutoTokenizer` with a SentencePiece model (e.g., `t5-small` or `xlm-roberta-base`).
   - **From Scratch:** Use `sentencepiece` module (`spm.SentencePieceTrainer.train`) to construct an unigram/bpe model natively and encode a sentence.
3. **Experiment 5.3 (POS Tagging on Sample):**
   - **spaCy:** Process the text through the `en_core_web_sm` pipeline. Yield `token.text`, `token.pos_`, and `spacy.explain(token.pos_)`.
   - **NLTK:** Tokenize using `word_tokenize`, tag using `pos_tag`, and optionally map tags to their standardized descriptions.
4. **Experiment 5.4 (POS Frequency on Corpus):**
   - Run spaCy over the entire `input_text.txt`.
   - Maintain a dictionary grouping by `(Token, POS, Description)` to aggregate frequency. Display the sorted frequencies.

---

## 4. Experimental Results and Observations

### Key Metrics Summary
| Experiment | Observation |
|:---|:---|
| 5.1 (BPE) | Pretrained models handle long words using a large vocabulary. Trained models segmented words heavily since the corpus is small. |
| 5.2 (SentencePiece) | Treated spaces as a character symbol `_`. Demonstrated similar structural slicing of morphemes. |
| 5.3 (POS Tagging) | NLTK uses Penn Treebank fine-grained tags (e.g., `VBG`), whereas spaCy outputs Universal POS tags (e.g., `VERB`). |

---

## 5. Error Analysis and Discussion
- **Subword Over-Segmentation:** When training BPE/SentencePiece from scratch on a small dataset (like `input_text.txt`), complex words that aren't in the text might be segmented into individual characters. This highlights the requirement for massive corpora when defining a universal vocabulary.
- **POS Tag Ambiguity:** Some words function as both verbs and nouns. Both spaCy and NLTK generally rely on local context to resolve such syntactic ambiguities efficiently.

---

## 6. Conclusion
- **Conclusion:** Subword tokenization represents a critical advancement for neural language models, striking the perfect balance between character-level flexibility and word-level semantic density. POS tagging further enriches this data, allowing downstream applications to discern grammar and structure autonomously.
