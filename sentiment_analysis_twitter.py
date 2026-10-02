#!/usr/bin/env python3
"""
Sentiment Analysis on Twitter Data
==================================
AI Internship Project - Codec Technologies

Classifies tweets as POSITIVE, NEGATIVE or NEUTRAL.

Approach
--------
1. Preprocessing : clean tweets (URLs, @handles, hashtags, punctuation, stop words),
                   tokenize (NLTK TweetTokenizer) and lemmatize (NLTK WordNet).
2. Model         : TF-IDF (unigrams + bigrams) + Logistic Regression.
3. Baseline      : NLTK VADER lexicon analyser (optional, used only if its lexicon
                   is available) so you can compare ML vs rule-based sentiment.
4. Evaluation    : accuracy, classification report, confusion matrix,
                   5-fold cross-validation, and the most influential words per class.

The script is fully self-contained: a hand-written sample dataset of tweets is
embedded below, so no API keys or downloads of tweet data are needed.

Setup
-----
    pip install numpy pandas scikit-learn matplotlib nltk
    python sentiment_analysis_twitter.py

NLTK resources (stopwords, wordnet, vader_lexicon) are downloaded automatically
on first run. If they cannot be downloaded (e.g. no internet), the script falls
back to scikit-learn stop words and skips lemmatization / VADER.
"""

from __future__ import annotations

import argparse
import re
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]


# =============================================================================
# 1. SAMPLE DATASET (synthetic, hand-written - no external data needed)
# =============================================================================
# Tweets deliberately include URLs, @handles, #hashtags, emoticons, emojis and
# contractions so that the preprocessing steps have real work to do.

POSITIVE_TWEETS = [
    "Just got my new laptop and it is absolutely amazing! Super fast and the battery lasts all day \U0001f60d #happy",
    "@AirIndia thank you for the smooth flight and the friendly crew, best journey I've had in years!",
    "Loved the movie tonight, the acting was brilliant and the ending made me tear up in a good way :)",
    "Best biryani in town!! Totally worth every rupee. Will definitely come back https://t.co/xyz123",
    "Woke up to great news today, I got the job! So grateful and excited for what's next \U0001f389",
    "The customer support team solved my problem in five minutes. Impressed and very satisfied.",
    "This new update is fantastic, the app feels so much smoother now. Great work devs! #appupdate",
    "What a beautiful sunrise this morning, feeling peaceful and thankful for everything",
    "Our team won the final! Absolutely thrilled, the crowd was incredible tonight \U0001f3c6",
    "I can't stop listening to this album, every single track is a masterpiece",
    "Had a wonderful weekend with family, good food, good music and lots of laughter :D",
    "Shoutout to @CafeBloom for the lovely service and the delicious coffee, my new favourite spot",
    "The online course is really well structured, I am learning so much and enjoying every lesson",
    "Delivery arrived a day early and everything was perfectly packed. Five stars!",
    "Proud of my sister for graduating with honours today. She worked so hard and deserves it \u2764\ufe0f",
    "Honestly this phone camera is stunning, the photos look gorgeous even at night",
    "Great concert last night! The band was energetic and the sound quality was superb",
    "Finally finished my first marathon and I feel unstoppable, what an amazing experience",
    "The hotel staff were so kind and helpful, the room was spotless and the view was breathtaking",
    "Love how easy this budgeting app is to use, it has genuinely made my life simpler",
    "Not bad at all, the new pizza place exceeded my expectations, crispy crust and fresh toppings",
    "Happy birthday to my best friend! Hope your day is full of joy and cake \U0001f382",
    "The new metro line is clean, fast and on time. Commuting has never been this pleasant",
    "Thank you to everyone who donated, we raised more than expected and it will help so many kids",
    "This book is a gem, beautifully written and impossible to put down",
    "Excellent service at the bank today, no waiting and the staff were polite and efficient",
    "I am so glad I tried yoga, I feel stronger, calmer and more positive every day",
    "The festival lights look magical this year, the whole city is glowing and cheerful",
    "Wow, the upgrade to the gym is brilliant, new equipment and a friendly trainer, highly recommend",
    "Perfect weather, great company and a delicious picnic. Today was a really good day",
]

NEGATIVE_TWEETS = [
    "Worst customer service ever. I waited two hours on hold and nobody solved my problem \U0001f621",
    "@FoodExpress my order arrived cold and half the items were missing. Totally unacceptable!",
    "This phone keeps crashing every hour, complete waste of money. Do not buy it",
    "So disappointed with the movie, the plot was boring and the acting was terrible :(",
    "The flight was delayed for five hours and the airline did not even apologise. Awful experience",
    "I hate how slow this app is, it freezes every time I open it #fail",
    "Terrible food and rude waiters. I will never go back to that restaurant again",
    "Internet has been down all day and the provider is not responding. So frustrating!!",
    "Feeling really sad and exhausted today, nothing seems to go right lately",
    "The hotel room was dirty, the AC was broken and the staff did not care at all",
    "Just paid a fortune for this and it broke after two days. Horrible quality",
    "Traffic is a nightmare again, stuck for two hours and I am so angry right now",
    "Can't believe the refund still has not arrived after a month. This is ridiculous and unprofessional",
    "The new update ruined everything, the design is ugly and half the features are gone",
    "My package got lost again. Useless delivery service, I am done with them https://t.co/abc999",
    "Awful concert, the sound was horrible and the crowd was pushing and shouting the whole time",
    "I regret buying this laptop, the screen flickers and the battery dies in an hour",
    "Disgusting service at the cafe today, found a hair in my food and they just shrugged",
    "This exam was brutal, I feel hopeless and I think I failed everything \U0001f622",
    "Another overpriced ticket and another disappointing match. Our defence is a joke",
    "Not happy at all with the repair, the problem is still there and they charged me full price",
    "The neighbours are being loud again at 2 AM, I am so tired and annoyed",
    "Worst mobile network, no signal at home and customer care never picks up the call",
    "Really let down by this book, the story drags and the ending makes no sense",
    "Nothing worked as promised, the product is a scam and the seller is not replying",
    "I am furious, they cancelled my booking without any notice and refused to compensate",
    "The queue at the bank was endless and the staff were incredibly rude. Never again",
    "So much garbage on the beach this morning, it is heartbreaking and shameful",
    "Poor packaging, damaged item and a useless support chat. Extremely unhappy",
    "Bad day, bad mood. Everything is going wrong and I just want to go home",
]

NEUTRAL_TWEETS = [
    "The meeting has been moved to 3 PM tomorrow in conference room B",
    "Just landed in Mumbai, heading to the hotel now",
    "New smartphone model will be launched on 14 October according to the company website",
    "Anyone know what time the library opens on Sundays?",
    "Reading a chapter on neural networks before my afternoon class #machinelearning",
    "The match starts at 7 PM and will be streamed live on the official channel",
    "Heavy rain expected in Bengaluru this evening, according to the weather department",
    "Submitted my assignment and waiting for the results to be announced next week",
    "The bus to the airport leaves every thirty minutes from platform four",
    "Our team is hosting a workshop on data science this Saturday, registration link in bio https://t.co/reg456",
    "Just had lunch and now heading back to the office",
    "The government announced a new policy on electric vehicles this morning",
    "Watching the news and having tea, nothing much going on today",
    "Update: the server will be down for maintenance between 1 AM and 3 AM",
    "The new cafe opens next to the metro station on the first of the month",
    "Is the exam syllabus the same as last year? Need to check with the professor",
    "Packing my bags for the trip, train departs at 6 in the morning",
    "Tomorrow is a public holiday so banks and offices will remain closed",
    "@priya_k I will send you the notes and the file by evening",
    "The conference has three tracks: machine learning, cloud and cybersecurity",
    "Changed my profile picture and updated my bio today",
    "Temperature today is 28 degrees with light winds from the east",
    "The documentary about ocean life airs on Sunday at 9 PM",
    "Picked up my parcel from the post office on the way home",
    "The company released its quarterly results and shares closed flat on Friday",
    "Learning Python and SQL this month, planning to start a small project next",
    "There will be a power cut in our area between 10 AM and noon on Wednesday",
    "Looking for a flatmate near the tech park, two bedroom apartment, rent shared",
    "The train from Chennai is running twenty minutes behind schedule",
    "Starting a new book today, a thriller recommended by a colleague",
]


def build_sample_dataset() -> pd.DataFrame:
    """Return a shuffled DataFrame with columns: text, sentiment."""
    df = pd.DataFrame(
        {
            "text": POSITIVE_TWEETS + NEGATIVE_TWEETS + NEUTRAL_TWEETS,
            "sentiment": (
                ["positive"] * len(POSITIVE_TWEETS)
                + ["negative"] * len(NEGATIVE_TWEETS)
                + ["neutral"] * len(NEUTRAL_TWEETS)
            ),
        }
    )
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# =============================================================================
# 2. NLP TOOLS (NLTK with graceful fallbacks)
# =============================================================================
class NLPTools:
    """Loads NLTK resources once; falls back to simpler tools if unavailable."""

    # Negations and intensifiers change sentiment, so they must NOT be removed
    KEEP_WORDS = {"no", "nor", "not", "never", "very", "too", "against"}

    def __init__(self) -> None:
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.tokenize: Callable[[str], List[str]] = str.split
        self.lemmatize: Callable[[str], str] = lambda w: w
        self.vader = None
        self.notes: List[str] = []
        self._setup()
        self.stop_words -= self.KEEP_WORDS

    def _setup(self) -> None:
        try:
            import nltk
        except ImportError:
            self.notes.append("NLTK not installed -> using fallbacks (pip install nltk).")
            return

        for pkg in ("stopwords", "wordnet", "omw-1.4", "vader_lexicon"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

        # Tokenizer (needs no downloaded data)
        try:
            from nltk.tokenize import TweetTokenizer

            tt = TweetTokenizer(preserve_case=False, reduce_len=True)
            self.tokenize = tt.tokenize
        except Exception:
            self.notes.append("TweetTokenizer unavailable -> using whitespace split.")

        # Stop words
        try:
            from nltk.corpus import stopwords

            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.notes.append("NLTK stop words unavailable -> using scikit-learn list.")

        # Lemmatizer
        try:
            from nltk.stem import WordNetLemmatizer

            lem = WordNetLemmatizer()
            lem.lemmatize("tests")  # raises LookupError if WordNet data is missing
            # verb pass first ("loved" -> "love"), then noun pass ("batteries" -> "battery")
            self.lemmatize = lambda w: lem.lemmatize(lem.lemmatize(w, "v"), "n")
        except Exception:
            self.notes.append("WordNet unavailable -> lemmatization skipped.")

        # VADER (optional baseline)
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            self.notes.append("VADER lexicon unavailable -> baseline comparison skipped.")


# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")          # keep the word, drop the '#'
RT_RE = re.compile(r"^\s*rt\s+")
REPEAT_RE = re.compile(r"(.)\1{2,}")        # "soooo" -> "soo"
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

CONTRACTIONS = [
    (r"won't", "will not"), (r"can't", "can not"), (r"n't", " not"),
    (r"'re", " are"), (r"'m", " am"), (r"'ll", " will"),
    (r"'ve", " have"), (r"'d", " would"),
]

# Emoticons / emojis carry strong sentiment, so map them to words BEFORE
# punctuation is stripped (otherwise that information is lost).
POS_EMOTICONS = [":)", ":-)", ":d", ":-d", ";)", "<3", "\U0001f60d", "\U0001f60a", "\U0001f600", "\U0001f601", "\U0001f389", "\u2764\ufe0f", "\u2764", "\U0001f44d", "\U0001f3c6", "\U0001f382"]
NEG_EMOTICONS = [":(", ":-(", ":'(", "\U0001f621", "\U0001f622", "\U0001f61e", "\U0001f620", "\U0001f44e", "\U0001f62d"]
EMOTICON_RE = re.compile(
    "|".join(re.escape(e) for e in sorted(POS_EMOTICONS + NEG_EMOTICONS, key=len, reverse=True))
)


def _emoticon_to_word(match: re.Match) -> str:
    return " happyface " if match.group(0) in POS_EMOTICONS else " sadface "


def clean_text(text: str) -> str:
    """Normalise a raw tweet into lowercase letters-only text."""
    text = str(text).lower().replace("\u2019", "'")
    text = RT_RE.sub("", text)                       # leading "RT"
    text = URL_RE.sub(" ", text)                     # URLs
    text = MENTION_RE.sub(" ", text)                 # @handles
    text = HASHTAG_RE.sub(r" \1 ", text)             # #happy -> happy
    text = EMOTICON_RE.sub(_emoticon_to_word, text)  # :) -> happyface
    for pattern, repl in CONTRACTIONS:               # don't -> do not
        text = re.sub(pattern, repl, text)
    text = REPEAT_RE.sub(r"\1\1", text)              # soooo -> soo
    text = NON_ALPHA_RE.sub(" ", text)               # punctuation, digits, leftovers
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str, nlp: NLPTools) -> str:
    """Full pipeline: clean -> tokenize -> remove stop words -> lemmatize."""
    tokens = nlp.tokenize(clean_text(text))
    tokens = [t for t in tokens if len(t) > 1 and t not in nlp.stop_words]
    tokens = [nlp.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# =============================================================================
# 4. MODEL
# =============================================================================
def build_model() -> Pipeline:
    """TF-IDF (1-2 grams) feeding a multinomial Logistic Regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ]
    )


def top_words_per_class(model: Pipeline, n: int = 6) -> dict:
    """Words with the largest positive weight for each class."""
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    clf = model.named_steps["clf"]
    return {cls: vocab[np.argsort(clf.coef_[i])[-n:][::-1]].tolist()
            for i, cls in enumerate(clf.classes_)}


# =============================================================================
# 5. EVALUATION
# =============================================================================
def vader_label(text: str, analyzer, threshold: float = 0.05) -> str:
    score = analyzer.polarity_scores(text)["compound"]
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"


def print_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                         columns=[f"pred_{l}" for l in LABELS])
    print(cm_df.to_string())


def save_confusion_matrix_plot(y_true, y_pred, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=LABELS, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("Tweet Sentiment - Confusion Matrix")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nConfusion matrix plot saved to: {path}")
    except ImportError:
        print("\n(matplotlib not installed - skipping confusion matrix plot)")


# =============================================================================
# 6. MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Twitter sentiment analysis (TF-IDF + Logistic Regression)")
    parser.add_argument("--data", help="Optional CSV with columns: text,sentiment (uses built-in sample otherwise)")
    parser.add_argument("--plot", default="confusion_matrix.png", help="Where to save the confusion matrix image")
    parser.add_argument("--predict", help="Classify a single tweet after training")
    # parse_known_args() ignores extra arguments injected by Jupyter/Colab,
    # so the script works both from the terminal and when pasted into a notebook cell.
    args, _ = parser.parse_known_args()

    nlp = NLPTools()
    for note in nlp.notes:
        print(f"NOTE: {note}")

    # ---- Load data -----------------------------------------------------------
    df = pd.read_csv(args.data) if args.data else build_sample_dataset()
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    print(f"\nDataset: {len(df)} tweets")
    print(df["sentiment"].value_counts().to_string())

    # ---- Preprocess ----------------------------------------------------------
    df["clean_text"] = df["text"].apply(lambda t: preprocess(t, nlp))
    print("\n--- Preprocessing example ---")
    sample = df.iloc[0]
    print(f"Original : {sample['text']}")
    print(f"Cleaned  : {sample['clean_text']}")

    # ---- Split ---------------------------------------------------------------
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        df["clean_text"], df["sentiment"], df["text"],
        test_size=0.25, random_state=RANDOM_STATE, stratify=df["sentiment"])

    # ---- Train ---------------------------------------------------------------
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---- Evaluate ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION + TF-IDF  (held-out test set)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix:")
    print_confusion_matrix(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_model(), df["clean_text"], df["sentiment"], cv=cv, scoring="accuracy")
    print(f"\n5-fold cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("\nMost influential words per class:")
    for cls, words in top_words_per_class(model).items():
        print(f"  {cls:<9}: {', '.join(words)}")

    save_confusion_matrix_plot(y_test, y_pred, args.plot)

    # ---- Optional VADER baseline --------------------------------------------
    if nlp.vader is not None:
        v_pred = [vader_label(t, nlp.vader) for t in raw_test]
        print("\n" + "=" * 60)
        print("VADER BASELINE (rule-based, no training)")
        print("=" * 60)
        print(f"Accuracy: {accuracy_score(y_test, v_pred):.3f}")

    # ---- Demo predictions ----------------------------------------------------
    print("\n--- Predictions on new tweets ---")
    demo = [
        "I absolutely love this new phone, the camera is gorgeous!",
        "Worst delivery experience ever, I am so upset \U0001f621",
        "The event starts at 5 PM tomorrow at the main hall",
        "This is not good at all, totally disappointed",
    ]
    if args.predict:
        demo.insert(0, args.predict)
    for tweet in demo:
        label = model.predict([preprocess(tweet, nlp)])[0]
        print(f"[{label.upper():<8}] {tweet}")


if __name__ == "__main__":
    main()
