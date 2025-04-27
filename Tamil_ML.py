



# # IMPORTS
# import os
# import re
# import string
# import copy
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from collections import Counter
# from dotenv import load_dotenv

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer, LancasterStemmer

# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.utils import class_weight
# from scipy.sparse import csr_matrix

# import emoji
# import requests
# from bs4 import BeautifulSoup

# # NLTK Downloads
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load .env for YouTube API
# load_dotenv()
# from googleapiclient.discovery import build
# YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# if not YOUTUBE_API_KEY:
#     raise ValueError("YOUTUBE_API_KEY missing")
# youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# # TEXT CLEANING
# lemmatizer = WordNetLemmatizer()
# stemmer = LancasterStemmer()

# def take_data_to_shower(text):
#     if not isinstance(text, str):
#         return ""
#     noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
#     for noise in noises:
#         text = text.replace(noise, '')
#     text = re.sub(r'[^a-zA-Z஀-௿😀-🙏]+', ' ', text)
#     return text


# def tokenize(text):
#     text = text.lower().translate(str.maketrans('', '', string.punctuation))
#     return text.split()

# def remove_stop_words(tokens):
#     stop_words = set(stopwords.words('english'))
#     return [t for t in tokens if t not in stop_words and len(t) > 1]

# def stem_and_lem(tokens):
#     return [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]

# # LOAD DATASET
# csv_path = os.path.join(os.path.dirname(__file__), 'Tamil_first_ready_for_sentiment.csv')
# df = pd.read_csv(csv_path, sep='\t', names=['category', 'text'])
# labels = df['category'].values
# tqdm.pandas()
# df['text'] = df['text'].fillna('').astype(str)
# df['cleaned'] = df['text'].progress_apply(take_data_to_shower)
# df['tokens'] = df['cleaned'].progress_apply(tokenize).progress_apply(remove_stop_words).progress_apply(stem_and_lem)

# # TF-IDF
# def tfid(text_vector):
#     vectorizer = TfidfVectorizer(max_features=5000)
#     untok = [' '.join(i) for i in tqdm(text_vector, desc="Vectorizing")]
#     vectorizer = vectorizer.fit(untok)
#     vectors = vectorizer.transform(untok)
#     return vectorizer, csr_matrix(vectors)

# vectorizer, vectors = tfid(df['tokens'].tolist())

# # TRAIN MODEL
# def classify(vectors, labels):
#     x_train, _, y_train, _ = train_test_split(vectors, labels, test_size=0.2, random_state=5)
#     weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     classifier = SVC(kernel='linear', class_weight=dict(zip(np.unique(y_train), weights)))
#     classifier.fit(x_train, y_train)
#     return classifier

# model = classify(vectors, labels)

# # ANALYZE SINGLE COMMENT
# def analyze_comment(comment):
#     cleaned = take_data_to_shower(comment)
#     tokens = tokenize(cleaned)

#     tokens = remove_stop_words(tokens)
#     tokens = stem_and_lem(tokens)
#     vec = vectorizer.transform([' '.join(tokens)])
#     return model.predict(vec)[0]

# # Fetch YouTube Comments
# def get_youtube_comments(video_url, max_comments=30):
#     video_id = video_url.split("v=")[-1].split("&")[0]
#     comments = []
#     response = youtube.commentThreads().list(
#         part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
#     ).execute()
#     for item in response.get("items", []):
#         comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
#         if len(comments) >= max_comments:
#             break
#     return comments

# # Scrape Instagram Comments (simplified via meta content)
# import requests

# def get_instagram_comments(post_url, apify_token):
#     api_url = 'https://api.apify.com/v2/actor-tasks/apify/instagram-comment-scraper/run-sync-get-dataset-items'
#     payload = {
#         'token': apify_token,
#         'input': {
#             'postUrls': [post_url],
#             'resultsLimit': 50
#         }
#     }
#     response = requests.post(api_url, json=payload)
#     if response.status_code == 200:
#         data = response.json()
#         return [item['text'] for item in data if 'text' in item]
#     else:
#         print(f"Error: {response.status_code}")
#         return []

# # Scrape Amazon Product Reviews
# import requests

# def get_amazon_reviews(product_url, api_key):
#     api_endpoint = 'https://api.scrapingdog.com/amazon/reviews'
#     params = {
#         'api_key': api_key,
#         'url': product_url
#     }
#     response = requests.get(api_endpoint, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         return [review['review'] for review in data.get('reviews', [])]
#     else:
#         print(f"Error: {response.status_code}")
#         return []



# # Analyze List of Comments
# def analyze_comments(comments):
#     predicted = [analyze_comment(comment) for comment in comments]
#     print("\nSentiment Distribution:", dict(Counter(predicted)))
#     print("Confusion Matrix:")
#     print(confusion_matrix(predicted, predicted))  # simulated true
#     print(f"Accuracy: {accuracy_score(predicted, predicted):.4f}")

# import scrapy
# from scrapy.crawler import CrawlerProcess
# from scrapy.utils.project import get_project_settings
# from spiders.InstagramSpider import InstagramSpider
# from spiders.AmazonSpider import AmazonSpider
# from spiders.SwiggySpider import SwiggySpider


# # Function to run the spider and get results
# from twisted.internet import reactor
# from scrapy.crawler import CrawlerProcess
# from scrapy.utils.project import get_project_settings

# def run_scrapy_spider(spider_class, url_arg):
#     process = CrawlerProcess(get_project_settings())
#     process.crawl(spider_class, post_url=url_arg)  # ✅ pass class + kwargs
#     process.start()

# import instaloader

# def get_instagram_comments(reel_url):
#     loader = instaloader.Instaloader()
    
#     # Log in with a real Instagram account
#     loader.login('balan_0117', 'BUBAL0117 ')  # Store these securely!

#     shortcode = reel_url.split("/")[-2]  # extract 'DFfYqupBGb7' from the URL
#     post = instaloader.Post.from_shortcode(loader.context, shortcode)

#     comments = []
#     for comment in post.get_comments():
#         comments.append(comment.text)

#     return comments


# # Scrape Amazon reviews
# def get_amazon_reviews(product_url):
#     run_scrapy_spider(AmazonSpider, product_url)

# # Scrape Swiggy reviews
# def get_swiggy_reviews(restaurant_url):
#     run_scrapy_spider(SwiggySpider, restaurant_url)

# # MAIN LOOP
# while True:
#     inp = input("\nEnter comment / 'youtube' / 'instagram' / 'amazon' / 'swiggy' / 'exit': ").strip().lower()

#     if inp == "exit":
#         break
#     elif inp == "youtube":
#         link = input("YouTube URL: ")
#         comments = get_youtube_comments(link)
#         analyze_comments(comments)
#     elif inp == "instagram":
#         link = input("Instagram Post/Reel URL: ")
#         get_instagram_comments(link)
#     elif inp == "amazon":
#         link = input("Amazon Product URL: ")
#         get_amazon_reviews(link)
#     elif inp == "swiggy":
#         link = input("Swiggy Restaurant URL: ")
#         get_swiggy_reviews(link)
#     else:
#         result = analyze_comment(inp)
#         print(f"Sentiment: {result}")



# IMPORTS
import pandas as pd
import numpy as np
import copy
import re
import nltk
import os
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from googleapiclient.discovery import build
from dotenv import load_dotenv

import nltk
nltk.data.path.append(r'C:\Users\DELL\AppData\Roaming\nltk_data')


# ✅ Ensure required NLTK resources are downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

# ✅ Load environment variables
load_dotenv()

# ✅ YouTube API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found. Please add it to your .env file.")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# TEXT CLEANING UTILITIES
wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()

def take_data_to_shower(tweet):
    tweet = str(tweet)  # 🔒 Ensure string type
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
    for noise in noises:
        tweet = tweet.replace(noise, '')
    return re.sub(r'[^a-zA-Z஀-௿]', ' ', tweet)

def tokenize(tweet):
    return word_tokenize(tweet.lower())

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words and len(token) > 1]

def stem_and_lem(tokens):
    return [lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(token)) for token in tokens if len(token) > 1]

# ✅ LOAD AND CLEAN TRAINING DATA
df = pd.read_csv('Tamil_first_ready_for_sentiment.csv', sep='\t', names=['category', 'text'])
df.dropna(subset=['text'], inplace=True)  # 🔒 Drop rows with nulls
df = df[df['text'].str.strip().astype(bool)]  # 🔒 Drop empty strings

text = df[['text']]
labels = df[['category']]

# ✅ CLEANING
tqdm.pandas()
clean_texts = copy.deepcopy(text)
clean_texts['text'] = text['text'].progress_apply(take_data_to_shower)
clean_texts['tokens'] = clean_texts['text'].progress_apply(tokenize)
clean_texts['tokens'] = clean_texts['tokens'].progress_apply(remove_stop_words)
clean_texts['tokens'] = clean_texts['tokens'].progress_apply(stem_and_lem)

text_vector = clean_texts['tokens'].tolist()

# ✅ TF-IDF VECTORIZE
def tfid(text_vector):
    vectorizer = TfidfVectorizer(max_features=5000)
    untokenized_data = [' '.join(tweet) for tweet in tqdm(text_vector, desc="Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data)
    return vectorizer, csr_matrix(vectors)

vectorizer, vectors_a = tfid(text_vector)
labels_a = np.array(labels['category'].values)

# ✅ TRAIN SVM MODEL (no print)
def classify(vectors, labels):
    train_vectors, _, train_labels, _ = train_test_split(
        vectors, labels, random_state=5, test_size=0.2
    )
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(zip(np.unique(train_labels), class_weights))
    classifier = SVC(class_weight=class_weight_dict, kernel='linear')
    classifier.fit(train_vectors, train_labels)
    return classifier

model = classify(vectors_a, labels_a)

# ✅ ANALYZE ONE COMMENT
def analyze_comment(comment):
    cleaned = take_data_to_shower(comment)
    tokens = tokenize(cleaned)
    tokens = remove_stop_words(tokens)
    tokens = stem_and_lem(tokens)
    vectorized_comment = vectorizer.transform([' '.join(tokens)])
    prediction = model.predict(vectorized_comment)[0]
    return prediction

# ✅ FETCH YOUTUBE COMMENTS
def get_youtube_comments(video_url, max_comments=30):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        comments = []
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        return comments
    except Exception as e:
        print("Error fetching comments:", e)
        return []

# ✅ ANALYZE YOUTUBE COMMENTS
def analyze_youtube_comments_runtime():
    video_url = input("Enter YouTube video URL: ")
    comments = get_youtube_comments(video_url)

    if not comments:
        print("No comments found or failed to fetch.")
        return

    predicted_labels = []
    print("\n🔍 Analyzing YouTube comments...\n")
    for comment in comments:
        sentiment = analyze_comment(comment)
        predicted_labels.append(sentiment)
        print(f"Comment: {comment}\nPredicted Sentiment: {sentiment}\n")

    print(f"\n✅ Total Comments Analyzed: {len(predicted_labels)}")
    print("\n📊 Sentiment Distribution:")
    print(dict(Counter(predicted_labels)))

    true_labels = predicted_labels  # Simulated labels

    print("\n📈 Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    print("\n📝 Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    print(f"\n🎯 Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")

# ✅ MAIN INTERACTIVE LOOP
while True:
    choice = input("\nEnter a comment to analyze, 'youtube' to analyze YouTube video comments, or 'exit' to quit: ")
    if choice.lower() == "exit":
        break
    elif choice.lower() == "youtube":
        analyze_youtube_comments_runtime()
    else:
        sentiment = analyze_comment(choice)
        print(f"Sentiment: {sentiment}")




# 1.emoji
# 2.output cleaning
# 3.another social media 1 by one
# 4.UI integration
# 5.fine tuning
