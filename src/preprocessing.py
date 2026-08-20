"""
Text preprocessing and normalization utilities for NLP experiments.
"""

from collections import Counter
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


def clean_text(
    text: str,
    lower: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
) -> str:
    """Cleans and standardizes raw text.

    Args:
        text (str): Input text string.
        lower (bool): If True, converts text to lowercase.
        remove_urls (bool): If True, strips web URLs.
        remove_html (bool): If True, strips HTML tags.
        remove_punctuation (bool): If True, removes all punctuation characters.
        remove_numbers (bool): If True, removes numeric digits.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)

    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    if remove_html:
        text = re.sub(r"<.*?>", "", text)

    if lower:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str, use_nltk: bool = True) -> List[str]:
    """Tokenizes text into a list of word tokens.

    Args:
        text (str): Input text.
        use_nltk (bool): If True and NLTK is installed, uses nltk.word_tokenize.
          Otherwise falls back to regex tokenization.

    Returns:
        List[str]: List of word tokens.
    """
    if use_nltk:
        try:
            import nltk
            return nltk.word_tokenize(text)
        except Exception:
            pass
    # Fallback to regex word tokenization
    return re.findall(r"\b\w+\b|[^\w\s]", text)


def tokenize_sentences(text: str, use_nltk: bool = True) -> List[str]:
    """Splits a body of text into individual sentences.

    Args:
        text (str): Input text.
        use_nltk (bool): If True and NLTK is installed, uses nltk.sent_tokenize.

    Returns:
        List[str]: List of sentence strings.
    """
    if use_nltk:
        try:
            import nltk
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    # Fallback regex sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def remove_stopwords(
    tokens: List[str],
    language: str = "english",
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """Filters out common stopwords from a list of tokens.

    Args:
        tokens (List[str]): List of word tokens.
        language (str): Language for NLTK stopword list (default: 'english').
        custom_stopwords (Set[str], optional): Additional custom stopwords to
          filter.

    Returns:
        List[str]: Filtered list of tokens.
    """
    stopwords_set: Set[str] = set()
    try:
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words(language))
    except Exception:
        # Default minimal fallback stopword set
        stopwords_set = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
        }

    if custom_stopwords:
        stopwords_set.update(custom_stopwords)

    return [t for t in tokens if t.lower() not in stopwords_set]


def stem_tokens(
    tokens: List[str], stemmer_type: str = "porter"
) -> List[str]:
    """Applies stemming to a list of word tokens.

    Args:
        tokens (List[str]): Tokens to stem.
        stemmer_type (str): 'porter' or 'snowball' or 'lancaster'.

    Returns:
        List[str]: Stemmed tokens.
    """
    try:
        from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

        if stemmer_type.lower() == "snowball":
            stemmer = SnowballStemmer("english")
        elif stemmer_type.lower() == "lancaster":
            stemmer = LancasterStemmer()
        else:
            stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
    except Exception:
        # Fallback if nltk stemmer is unavailable
        return tokens


def lemmatize_tokens(
    tokens: List[str], pos_tags: Optional[List[str]] = None
) -> List[str]:
    """Applies WordNet lemmatization to a list of word tokens.

    Args:
        tokens (List[str]): Tokens to lemmatize.
        pos_tags (List[str], optional): Corresponding POS tags (Penn Treebank or
          WordNet format).

    Returns:
        List[str]: Lemmatized tokens.
    """
    try:
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        def _penn2wordnet(tag: str) -> str:
            if tag.startswith("J"):
                return wordnet.ADJ
            elif tag.startswith("V"):
                return wordnet.VERB
            elif tag.startswith("N"):
                return wordnet.NOUN
            elif tag.startswith("R"):
                return wordnet.ADV
            return wordnet.NOUN

        if pos_tags and len(pos_tags) == len(tokens):
            return [
                lemmatizer.lemmatize(tok, _penn2wordnet(tag))
                for tok, tag in zip(tokens, pos_tags)
            ]
        return [lemmatizer.lemmatize(tok) for tok in tokens]
    except Exception:
        return tokens


def generate_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generates contiguous sequences of n-items from a token sequence.

    Args:
        tokens (List[str]): Sequence of tokens.
        n (int): Size of the n-gram (e.g. 2 for bigram, 3 for trigram).

    Returns:
        List[Tuple[str, ...]]: List of n-gram tuples.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_vocabulary(
    tokenized_corpus: Iterable[List[str]],
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
    bos_token: Optional[str] = "<BOS>",
    eos_token: Optional[str] = "<EOS>",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Builds word-to-index and index-to-word mappings from a tokenized corpus.

    Args:
        tokenized_corpus (Iterable[List[str]]): List of token lists.
        min_freq (int): Minimum occurrence threshold to include in vocabulary.
        max_vocab_size (int, optional): Maximum vocabulary size limit.
        pad_token (str): Padding special token.
        unk_token (str): Out-of-vocabulary unknown token.
        bos_token (str, optional): Beginning-of-sequence token.
        eos_token (str, optional): End-of-sequence token.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: (word2idx, idx2word)
    """
    counts = Counter()
    for tokens in tokenized_corpus:
        counts.update(tokens)

    special_tokens = [tok for tok in [pad_token, unk_token, bos_token, eos_token] if tok]
    word2idx: Dict[str, int] = {tok: idx for idx, tok in enumerate(special_tokens)}

    sorted_words = [
        word for word, count in counts.most_common()
        if count >= min_freq and word not in word2idx
    ]

    if max_vocab_size:
        remaining_slots = max_vocab_size - len(word2idx)
        sorted_words = sorted_words[:remaining_slots]

    for word in sorted_words:
        word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word
