import requests
import plotly.graph_objects as go
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models
from collections import Counter

headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}

# Function to clean and preprocess text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to extract text after stars
def extract_text_after_stars(text):
    match = re.search(r'(?<=stars\s).*', text)
    return match.group(0).strip() if match else text

# Function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Function to extract HTML data from Amazon Review page
def reviews_html(url, len_page):
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

# Function to extract reviews' name, description, date, stars, title from HTML
def get_reviews(html_data):
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')
    for box in boxes:
        name = box.select_one('[class="a-profile-name"]').text.strip() if box.select_one('[class="a-profile-name"]') else 'N/A'
        stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A'
        title = box.select_one('[data-hook="review-title"]').text.strip() if box.select_one('[data-hook="review-title"]') else 'N/A'
        datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1] if box.select_one('[data-hook="review-date"]') else 'N/A'
        date = datetime.strptime(datetime_str, '%d %B %Y').strftime("%d/%m/%Y") if datetime_str != 'N/A' else 'N/A'
        description = box.select_one('[data-hook="review-body"]').text.strip() if box.select_one('[data-hook="review-body"]') else 'N/A'

        data_dict = {
            'Name': name,
            'Stars': stars,
            'Title': title,
            'Date': date,
            'Description': description
        }
        data_dicts.append(data_dict)
    return data_dicts

# Function to generate word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.title(f'Word Cloud for {title}')
    st.image(wordcloud.to_array())

# Function to generate bar chart
def generate_bar_chart(text, title):
    # Use Counter to count the occurrences of each word
    word_counts = Counter(text.split())

    # Extract the most common words and their frequencies
    top_words = word_counts.most_common(10)

    # Plotting a bar chart
    st.write(f"Top 10 Most Common Words in {title}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([word[0] for word in top_words], [count[1] for count in top_words], color='skyblue')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Top 10 Most Common Words in {title}')
    plt.xticks(rotation=45, ha='right')  # Rotate
    st.pyplot(fig)

# Streamlit app
st.title("Amazon Product Review Analysis")

# User input for Amazon product review URL
reviews_url = st.text_input("Enter Amazon Product Review URL:")
if not reviews_url:
    st.warning("Please enter a valid Amazon Product Review URL.")
    st.stop()

# Define the number of pages
len_page = st.slider("Select the number of pages to scrape:", min_value=1, max_value=20, value=10)

# Fetch and process reviews data
html_datas = reviews_html(reviews_url, len_page)
reviews_data = []

for html_data in html_datas:
    review_data = get_reviews(html_data)
    reviews_data += review_data

df_reviews = pd.DataFrame(reviews_data)

# Display the reviews DataFrame
st.subheader("Reviews Data:")
st.dataframe(df_reviews)

# Clean data
df_reviews['Title'] = df_reviews['Title'].apply(extract_text_after_stars)
df_reviews['Title'] = df_reviews['Title'].apply(clean_text)
df_reviews['Description'] = df_reviews['Description'].apply(clean_text)

# Give sentiment
df_reviews['Title_Sentiment'] = df_reviews['Title'].apply(get_sentiment)
df_reviews['Description_Sentiment'] = df_reviews['Description'].apply(get_sentiment)
df_reviews['Title_Sentiment_Label'] = df_reviews['Title_Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
df_reviews['Description_Sentiment_Label'] = df_reviews['Description_Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Visualize sentiment distribution in Title
sentiment_counts_title = df_reviews['Title_Sentiment_Label'].value_counts()
fig_title = go.Figure(data=[go.Pie(labels=sentiment_counts_title.index, values=sentiment_counts_title)])
fig_title.update_layout(title='Sentiment Distribution in Title')
st.plotly_chart(fig_title)

# Visualize sentiment distribution in Description
sentiment_counts_description = df_reviews['Description_Sentiment_Label'].value_counts()
fig_description = go.Figure(data=[go.Pie(labels=sentiment_counts_description.index, values=sentiment_counts_description)])
fig_description.update_layout(title='Sentiment Distribution in Description')
st.plotly_chart(fig_description)

# Generate word cloud for Title
generate_word_cloud(' '.join(df_reviews['Title'].astype(str)), 'Title')

# Generate word cloud for Description
generate_word_cloud(' '.join(df_reviews['Description'].astype(str)), 'Description')

# Generate bar chart for Title and Description
generate_bar_chart(' '.join(df_reviews['Title'].astype(str)), 'Title')
generate_bar_chart(' '.join(df_reviews['Description'].astype(str)), 'Description')

# Tokenize the combined text column, ensuring all inputs are strings
tokenized_text = [word_tokenize(str(doc)) for doc in df_reviews['Title'] + ' ' + df_reviews['Description']]

# Create a Dictionary from the articles: dictionary
dictionary = corpora.Dictionary(tokenized_text)

# Create a Corpus: corpus
corpus = [dictionary.doc2bow(text) for text in tokenized_text]

# Fit the LDA model on the corpus
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)

# Print the 5 topics with their 10 most significant words
topics = lda_model.print_topics(num_words=10)
 

# Visualize the LDA model
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
# Save the visualization to an HTML file
pyLDAvis.save_html(lda_vis, 'lda_vis.html')

# Read the HTML content
with open("lda_vis.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Display the PyLDAvis visualization using st.components.v1.html
st.components.v1.html(html_content, height=800,width=900 ,scrolling=True)

