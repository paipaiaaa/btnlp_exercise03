# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import spacy

from textblob import TextBlob
import re
from summarizer import Summarizer

# Step 1: Read in the twitter_training.csv
df = pd.read_csv('/content/twitter_training.csv')

# Step 2: Run summary/descriptive statistics tests on the data
pre_df = df.copy()  # Rename this dataframe to pre_df
print("Head of the DataFrame:")
print(pre_df.head())
print("\nSummary Statistics:")
print(pre_df.describe())

# Step 3: Text transformations
# - Determine if there are missing values
print("\nMissing Values:")
print(pre_df.isnull().sum())

# Import necessary libraries for BERT summarization
!pip install gensim
from summarizer import Summarizer

# Initialize BERT summarizer
bert_model = Summarizer()


# Assuming the text is in the first column (index 0)
pre_df['summary'] = pre_df.iloc[:, 0].astype(str).apply(lambda x: bert_model(x))


# Display the summarized text
print("\nBERT-Extractive Summarization:")
print(pre_df['summary'].head())



# Assuming the target column is at index 0
pre_df.iloc[:, 0] = pre_df.iloc[:, 0].astype(str).str.lower()
pre_df.iloc[:, 0] = pre_df.iloc[:, 0].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
pre_df.iloc[:, 0] = pre_df.iloc[:, 0].apply(lambda x: ' '.join(x.split()))
pre_df['word_list'] = pre_df.iloc[:, 0].apply(lambda x: word_tokenize(x))



# Step 4: Apply features from various libraries
# - Scikit Learn (CountVectorizer)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(pre_df.iloc[:, 0])
print("\nScikit Learn Feature (CountVectorizer):")
print(X.toarray())




# - Pandas (Word frequency)
word_freq = pre_df['word_list'].apply(lambda x: pd.value_counts(x)).sum(axis=0).sort_values(ascending=False)
print("\nPandas Feature (Word Frequency):")
print(word_freq)

# - Matplotlib (Word cloud)
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# - Regex (Extract hashtags)
pre_df['hashtags'] = pre_df.iloc[:, 0].apply(lambda x: re.findall(r'#\w+', x))
print("\nRegex Feature (Extracted Hashtags):")
print(pre_df['hashtags'].head())

# Step 5: Apply features from NLP libraries
# - NLTK (Word frequency distribution)
all_words = [word for sublist in pre_df['word_list'] for word in sublist]
fdist = FreqDist(all_words)
print("\nNLTK Feature (Word Frequency Distribution):")
print(fdist)

# - SpaCy (Named Entity Recognition)
nlp = spacy.load('en_core_web_sm')
pre_df['ner'] = pre_df.iloc[:, 0].apply(lambda x: [(ent.text, ent.label_) for ent in nlp(x).ents])
print("\nSpaCy Feature (Named Entity Recognition):")
print(pre_df['ner'].head())

# - Gensim (Keyword extraction)
pre_df['keywords'] = pre_df.iloc[:, 0].apply(lambda x: keywords(x, words=5))
print("\nGensim Feature (Keyword Extraction):")
print(pre_df['keywords'].head())

# - TextBlob (Sentiment Analysis)
pre_df['sentiment'] = pre_df.iloc[:, 0].apply(lambda x: TextBlob(x).sentiment.polarity)
print("\nTextBlob Feature (Sentiment Analysis):")
print(pre_df['sentiment'].head())




import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class TextProcessor:
    def __init__(self, text):
        self.text = text

    def lowercase(self):
        try:
            self.text = self.text.lower()
        except AttributeError:
            print("Input must be a string.")

    def remove_non_ascii(self):
        try:
            self.text = re.sub(r'[^\x00-\x7F]+', ' ', self.text)
        except AttributeError:
            print("Input must be a string.")

    def remove_extra_whitespace(self):
        try:
            self.text = ' '.join(self.text.split())
        except AttributeError:
            print("Input must be a string.")

    def tokenize_text(self):
        try:
            self.word_list = word_tokenize(self.text)
        except AttributeError:
            print("Input must be a string.")

    def word_frequency_analysis(self):
        try:
            all_words = [word for word in self.word_list]
            fdist = FreqDist(all_words)
            return fdist
        except AttributeError:
            print("Text must be tokenized first.")





# Exercise_8:Although I'm 2nd year student already, I still did a lot of things wrong when I was doing the homework. The exercise 6 is very complex for me, and to those who just started programming it could be even more difficult.I also feel like this part is a bit overlapped with the introduction to python course.