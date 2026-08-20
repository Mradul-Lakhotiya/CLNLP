# Experiment 02: Basic Text Preprocessing

**Course:** Computational Linguistics and Natural Language Processing (CLNLP) Lab  
**Experiment Name:** Experiment-2: Basic Text Preprocessing
**Date:** 2026-08-20  

---

## 1. Aim and Objectives
- **Aim:** To understand and implement fundamental text preprocessing techniques required for downstream Natural Language Processing tasks.
- **Objectives:**
  - [ ] Objective 1: Implement basic text cleaning operations including lowercasing, punctuation removal, number removal, and whitespace normalization.
  - [ ] Objective 2: Perform word and sentence tokenization using custom python logic, NLTK, and spaCy libraries.
  - [ ] Objective 3: Remove stop words from a given text to reduce dimensionality and focus on important words.

---

## 2. Theoretical Background and Linguistic Concepts
- **Lowercasing:** Reduces the vocabulary size by treating different capitalizations of the same word as identical (e.g., "Apple" and "apple").
- **Punctuation and Number Removal:** Removes noise from the text, as these often do not carry semantic meaning in traditional NLP tasks.
- **Tokenization:** The process of breaking down a continuous stream of text into meaningful units called tokens (words, subwords, or sentences).
- **Stop Word Removal:** Eliminates highly frequent words (e.g., "the", "is", "in") that do not contribute significant meaning to the sentence context.

---

## 3. Algorithm and Methodology
1. **Corpus Selection / Input Preparation:** Use sample text files containing raw text data.
2. **Experiment 2.1 (Basic Preprocessing):** Read text, compute uppercase counts, lowercase the text, remove punctuation using `string.punctuation`, remove digits using regex `\d`, and remove extra whitespace.
3. **Experiment 2.2 (Tokenization):** Apply custom regex-based tokenization, NLTK's `word_tokenize` and `sent_tokenize`, and spaCy's linguistic models for word and sentence tokenization.
4. **Experiment 2.3 (Stop Word Removal):** Tokenize the text and filter out words present in NLTK's English stopwords list.

---

## 4. Experimental Results and Observations

### Key Metrics Summary
| Technique | Output Observation |
|:---|:---|
| Basic Cleaning | Reduced uppercase letters, removed numbers, normalized spacing. |
| Tokenization (NLTK vs spaCy) | SpaCy handles complex punctuation slightly differently than NLTK. |
| Stop Word Removal | Token count significantly reduced, leaving only content words. |

---

## 5. Error Analysis and Discussion
- Tokenization using simple split methods fails to separate punctuation attached to words.
- Context-specific stop words might be accidentally removed (e.g., "Will" as a name vs modal verb).
- Lemmatization would be a logical next step to further normalize tokens to their dictionary forms.

---

## 6. Conclusion
- **Conclusion:** The fundamental text preprocessing pipeline was successfully implemented. Tokenization and stopword removal significantly reduce vocabulary size, preparing the text for feature extraction and modeling.
