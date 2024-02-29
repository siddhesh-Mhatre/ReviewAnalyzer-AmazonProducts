 
 
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

 
headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}

# URL of the Amazon Review page
reviews_url = 'https://www.amazon.in/product-reviews/B07JL3W3KG/'

# Define the number of pages
len_page = 10

# Functions

# Extract HTML data from Amazon Review page
def reviewsHtml(url, len_page):
    soups = []

    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }

        response = requests.get(url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, 'lxml')
        soups.append(soup)

    return soups

# Extract reviews' name, description, date, stars, title from HTML
def getReviews(html_data):
    data_dicts = []

    boxes = html_data.select('div[data-hook="review"]')

    for box in boxes:
        try:
            name = box.select_one('[class="a-profile-name"]').text.strip()
        except Exception as e:
            name = 'N/A'

        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
        except Exception as e:
            stars = 'N/A'

        try:
            title = box.select_one('[data-hook="review-title"]').text.strip()
        except Exception as e:
            title = 'N/A'

        try:
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'

        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'

        data_dict = {
            'Name' : name,
            'Stars' : stars,
            'Title' : title,
            'Date' : date,
            'Description' : description
        }

        data_dicts.append(data_dict)

    return data_dicts

# Data Process

html_datas = reviewsHtml(reviews_url, len_page)
reviews = []

for html_data in html_datas:
    review = getReviews(html_data)
    reviews += review

df_reviews = pd.DataFrame(reviews)
print(df_reviews)

# Save data
# df_reviews.to_csv('reviews.csv', index=False)

import re

df= df_reviews
def extract_text_after_stars(text):
    match = re.search(r'(?<=stars\s).*', text)
    return match.group(0).strip() if match else text

# Apply the function to the 'Title' column
df['Title'] = df['Title'].apply(extract_text_after_stars)

"""# cleaning the (Title , Description)"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

 

# Load the dataset

# Function to clean and preprocess text
def clean_text(text):
    # Ensure text is a string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Apply the cleaning function to 'Title' and 'Description' columns
df['Title'] = df['Title'].apply(clean_text)
df['Description'] = df['Description'].apply(clean_text)

"""# give sentiment"""

 

import pandas as pd
from textblob import TextBlob

# Assuming you have a dataset named 'your_dataset' with columns: Name, Stars, Title, Date, Description

# Function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Applying the sentiment analysis function to the 'Title' and 'Description' columns
df['Title_Sentiment'] = df['Title'].apply(get_sentiment)
df['Description_Sentiment'] = df['Description'].apply(get_sentiment)

# Creating a new column for sentiment labels based on polarity
df['Title_Sentiment_Label'] = df['Title_Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
df['Description_Sentiment_Label'] = df['Description_Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

import matplotlib.pyplot as plt

# Assuming you have a dataset named 'your_dataset' with Title_Sentiment_Label column

# Count the number of occurrences for each sentiment label
sentiment_counts = df['Title_Sentiment_Label'].value_counts()

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Sentiment Distribution in Title')
plt.show()

# Count the number of occurrences for each sentiment label
sentiment_counts = df['Description_Sentiment_Label'].value_counts()

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Sentiment Distribution in Title')
plt.show()
 

from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Assuming you have a dataset named 'your_dataset' with a 'Title' column

# Combine all titles into a single string
all_titles = ' '.join(df['Title'].astype(str))

# Use Counter to count the occurrences of each word
word_counts = Counter(all_titles.split())

# Extract the most common words and their frequencies
top_words = word_counts.most_common(10)

# Plotting a bar chart
plt.figure(figsize=(10, 6))
plt.bar([word[0] for word in top_words], [count[1] for count in top_words], color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in Titles')
plt.xticks(rotation=45, ha='right')  # Rotate

from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Assuming you have a dataset named 'df' with a 'Title' column

# Combine all titles into a single string
all_titles = ' '.join(df['Description'].astype(str))

# Use Counter to count the occurrences of each word
word_counts = Counter(all_titles.split())

# Extract the most common words and their frequencies
top_words = word_counts.most_common(10)

# Plotting a bar chart
plt.figure(figsize=(10, 6))
plt.bar([word[0] for word in top_words], [count[1] for count in top_words], color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in Titles')
plt.xticks(rotation=45, ha='right')  # Rotate

 

import nltk
 
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Tokenize the combined text column, ensuring all inputs are strings
tokenized_text = [word_tokenize(str(doc)) for doc in df['Title'] + ' ' + df['Description']]

# Create a Dictionary from the articles: dictionary
dictionary = corpora.Dictionary(tokenized_text)

# Create a Corpus: corpus
corpus = [dictionary.doc2bow(text) for text in tokenized_text]

# Fit the LDA model on the corpus
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)

# Print the 5 topics with their 10 most significant words
topics = lda_model.print_topics(num_words=10)
topics

 

import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel
from gensim import corpora

# Assuming 'lda_model', 'corpus', and 'dictionary' are already defined in your code
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# Save the visualization to an HTML file
pyLDAvis.display(lda_vis)

