# Amazon Product Review Analysis

This project aims to analyze and visualize Amazon product reviews using various natural language processing (NLP) techniques. It fetches reviews data from an Amazon product review URL, processes the text, and provides insights through visualizations.

## Technologies Used
- Python
- Requests
- Plotly
- Pandas
- BeautifulSoup
- NLTK
- TextBlob
- Streamlit
- Matplotlib
- WordCloud
- Gensim

## Overview
The project fetches Amazon product reviews from a specified URL and extracts relevant information such as reviewer name, review date, star rating, review title, and description. It then performs data cleaning and preprocessing steps including text normalization, sentiment analysis, and visualization of sentiment distribution.

## Features
- Fetches reviews data from Amazon product review URL
- Cleans and preprocesses text data (removes punctuation, stopwords, etc.)
- Analyzes sentiment polarity of review titles and descriptions
- Visualizes sentiment distribution using pie charts
- Generates word clouds to visualize most common words in review titles and descriptions
- Creates bar charts to display frequency of top words in review titles and descriptions
- Applies Latent Dirichlet Allocation (LDA) topic modeling to identify themes/topics in the reviews
- Visualizes LDA topic modeling results using PyLDAvis

## How to Use
1. Run the Streamlit app.
2. Enter a valid Amazon product review URL.
3. Adjust the number of pages to scrape (optional).
4. View the extracted reviews data and sentiment analysis visualizations.
5. Explore word clouds, bar charts, and LDA topic modeling results for deeper insights.

## Example Usage
- Analyze sentiment distribution to understand overall sentiment towards a product.
- Explore word clouds and bar charts to identify common themes and frequently mentioned keywords.
- Use LDA topic modeling to uncover underlying topics or trends in the reviews.

## Future Improvements
- Implement user authentication for accessing Amazon product review data.
- Add support for analyzing reviews from multiple sources (e.g., other e-commerce platforms, social media).
- Enhance text preprocessing techniques for better analysis and insights.
- Improve visualization aesthetics and interactivity for a more engaging user experience.

## Acknowledgments
- This project was inspired by the need for understanding customer sentiments and opinions on e-commerce platforms.
- Special thanks to the open-source community for developing and maintaining the libraries used in this project.

## Demo
- ![image](https://github.com/siddhesh-Mhatre/ReviewAnalyzer-AmazonProducts/assets/80941193/e3390774-4294-4045-97f3-00268c7af754)
- ![image](https://github.com/siddhesh-Mhatre/ReviewAnalyzer-AmazonProducts/assets/80941193/d955a94c-ebe6-42e7-85bd-f3886c4718f1)
- ![image](https://github.com/siddhesh-Mhatre/ReviewAnalyzer-AmazonProducts/assets/80941193/24a14694-aeed-4c1d-9f2a-abc815c3fa6c)
- ![image](https://github.com/siddhesh-Mhatre/ReviewAnalyzer-AmazonProducts/assets/80941193/56ae3fe8-3210-46a7-9b23-a5782f182e12)
- ![image](https://github.com/siddhesh-Mhatre/ReviewAnalyzer-AmazonProducts/assets/80941193/1366292d-55b1-4678-9990-5363b4f77b02)
- ![image](https://github.com/siddhesh-Mhatre/ReviewAnalyzer-AmazonProducts/assets/80941193/bd60dba1-7582-43e1-9ab7-8892c8b4bc0e)







Feel free to contribute or provide feedback to improve this project!

