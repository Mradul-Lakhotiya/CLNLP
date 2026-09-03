# Experiment 04: Term-Frequency, NER, and TF-IDF Analysis

**Course:** Computational Linguistics and Natural Language Processing (CLNLP) Lab  
**Experiment Name:** Experiment-4: Term-Frequency, NER, and TF-IDF Analysis  
**Date:** 2026-08-27  

---

## 1. Aim and Objectives
- **Aim:** To compute term frequencies, extract Named Entities using toolkits, and perform a complete TF-IDF calculation from scratch to analyze term importance across documents.
- **Objectives:**
  - [ ] Objective 1 (Exp 4.1): Perform Term-Frequency Analysis and Named Entity Recognition (NER) on a given text using a toolkit (e.g., NLTK/spaCy), save frequencies to CSV, and display top 10 terms.
  - [ ] Objective 2 (Exp 4.2): Perform Term-Frequency Analysis on a given text *without* using any NLP toolkit, save to CSV, and display top 10 terms.
  - [ ] Objective 3 (Exp 4.3): Implement TF-IDF (Term Frequency-Inverse Document Frequency) computation entirely from scratch for multiple documents, outputting TF, DF, IDF, and TF-IDF tables, and highlighting the top 10 terms per document.

---

## 2. Theoretical Background and Linguistic Concepts
- **Term Frequency (TF):** The raw count of a term in a document. High frequency may imply high relevance, but often stop words dominate this metric.
- **Named Entity Recognition (NER):** A subtask of information extraction that classifies named entities in unstructured text into predefined categories such as person names, organizations, locations, etc.
- **Document Frequency (DF) & Inverse Document Frequency (IDF):** DF measures the number of documents containing a term. IDF is the logarithmically scaled inverse fraction of the documents that contain the word ($IDF = \log(N / DF)$). It penalizes commonly occurring words across the corpus.
- **TF-IDF:** The product of TF and IDF ($TF \times IDF$). This score evaluates how important a word is to a document within a collection or corpus.

---

## 3. Algorithm and Methodology
1. **Experiment 4.1 (TF & NER with Toolkit):** 
   - Load text and tokenize using NLTK/spaCy.
   - Use `collections.Counter` to find term frequencies.
   - Run spaCy's `en_core_web_sm` model to extract entities (`doc.ents`).
   - Save frequencies to `outputs/4_1_tf.csv` using Pandas.
2. **Experiment 4.2 (TF without Toolkit):** 
   - Use basic Python string operations (`lower()`, `translate()`) to clean text.
   - Split by whitespace to tokenize.
   - Build a frequency dictionary manually.
   - Save frequencies to `outputs/4_2_tf.csv` using the `csv` module.
3. **Experiment 4.3 (TF-IDF from Scratch):** 
   - Process three documents: lowercase, remove punctuation, and split into tokens.
   - **TF:** Count occurrences of each term per document.
   - **DF:** Count how many documents contain each term at least once.
   - **IDF:** Calculate $\log_{10}(\frac{N}{DF})$ where $N=3$.
   - **TF-IDF:** Multiply $TF \times IDF$.
   - Sort and extract the top 10 TF-IDF terms per document.

---

## 4. Experimental Results and Observations

### Key Metrics Summary
| Experiment | Observation |
|:---|:---|
| 4.1 (Toolkit) | TF efficiently ignores casing if preprocessed. NER successfully identifies abstract concepts (like "NLP") and languages. |
| 4.2 (No Toolkit) | Achieved identical TF results as the toolkit approach, demonstrating basic string manipulation logic. |
| 4.3 (TF-IDF) | Stop words (e.g., "is", "of") receive high TF but very low IDF (often 0), causing their TF-IDF to drop. Domain-specific words receive the highest TF-IDF. |

---

## 5. Error Analysis and Discussion
- **Tokenization differences:** Manual splitting by space might not accurately decouple punctuation attached to words without rigorous regex, whereas NLTK/spaCy handle sub-word punctuation efficiently.
- **Log Base:** TF-IDF values vary slightly depending on whether $\log_e$ (natural log) or $\log_{10}$ is used. $\log_{10}$ was selected for interpretability.
- **Zero Division Risk:** If $DF=0$, IDF throws an error, but this is theoretically impossible for terms extracted directly from the corpus.

---

## 6. Conclusion
- **Conclusion:** TF-IDF effectively discounts the influence of highly frequent stop words that TF alone would falsely prioritize. Relying solely on Python standard libraries gives profound insight into the algorithmic mechanics hidden behind toolkits like NLTK or scikit-learn.
