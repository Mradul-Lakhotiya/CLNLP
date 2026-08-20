# Experiment 03: Stemming, Lemmatization and Regular Expression

**Course:** Computational Linguistics and Natural Language Processing (CLNLP) Lab  
**Experiment Name:** Experiment-3: Stemming, Lemmatization and Regular Expression  
**Date:** 2026-08-20  

---

## 1. Aim and Objectives
- **Aim:** To implement and analyze morphological normalization techniques (Stemming and Lemmatization) and extract structured information from raw text using Regular Expressions (Regex).
- **Objectives:**
  - [ ] Objective 1: Perform stemming using the Porter Stemmer on a predefined list of words and compare the stems with original words.
  - [ ] Objective 2: Perform lemmatization on a predefined list of words to reduce them to their base dictionary forms.
  - [ ] Objective 3: Utilize Regular Expressions (Regex) to extract Email addresses, URLs, Mobile numbers, Hashtags, and Mentions from a given textual dataset.

---

## 2. Theoretical Background and Linguistic Concepts
- **Stemming:** A heuristic process that drops the ends of words in the hope of achieving the goal correctly most of the time. It often results in non-dictionary words (e.g., "studying" -> "studi"). The Porter Stemmer is a widely used algorithm for English.
- **Lemmatization:** A more sophisticated approach that uses vocabulary and morphological analysis to return the base or dictionary form of a word, known as the lemma (e.g., "better" -> "good", "mice" -> "mouse").
- **Regular Expressions (Regex):** A sequence of characters that specifies a search pattern in text. It is a powerful tool for information extraction, such as identifying standard formats for emails, URLs, or phone numbers.

---

## 3. Algorithm and Methodology
1. **Experiment 3.1 (Stemming):** Initialize NLTK's `PorterStemmer`. Iterate over the given word list (`playing, played, plays, studies, studying, connected, connection, computers`) and apply the stemmer.
2. **Experiment 3.2 (Lemmatization):** Initialize NLTK's `WordNetLemmatizer`. Iterate over the word list (`cats, dogs, running, runs, ran, studies, studying, better, children, mice, went, ate, leaves, caring`). Apply the lemmatizer, employing Part-of-Speech (POS) tagging where necessary for accurate lemmatization.
3. **Experiment 3.3 (Regex Extraction):** Define Regex patterns for:
   - **Emails:** `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
   - **URLs:** `https?://[^\s]+|www\.[^\s]+`
   - **Mobile numbers:** `\+?\d{1,3}[-.\s]?\d{10}|\b\d{10}\b`
   - **Hashtags:** `#[a-zA-Z0-9_]+`
   - **Mentions:** `@[a-zA-Z0-9_]+` (excluding emails)
   Apply `re.findall()` on the input text to extract matches.

---

## 4. Experimental Results and Observations

### Key Metrics Summary
| Technique | Observation |
|:---|:---|
| Stemming | Successfully chops suffixes but sometimes produces non-dictionary words (e.g., "studies" -> "studi"). |
| Lemmatization | Accurately reduces words to dictionary forms (e.g., "mice" -> "mouse"), provided the correct POS tag is applied. |
| Regex Extraction | Accurately parsed semi-structured data points from unstructured paragraph text. |

---

## 5. Error Analysis and Discussion
- **Stemming vs Lemmatization:** Stemming is faster but less accurate, leading to over-stemming or under-stemming. Lemmatization requires POS tags to correctly identify verbs vs nouns (e.g., "running" as a noun vs verb).
- **Regex Limitations:** Regular expressions can be brittle. An overly restrictive regex might miss valid email formats, while an overly permissive one might capture invalid strings. 

---

## 6. Conclusion
- **Conclusion:** Both stemming and lemmatization effectively reduce vocabulary size. Lemmatization provides higher linguistic accuracy. Regular expressions remain an indispensable tool for targeted information extraction in natural language processing pipelines.
