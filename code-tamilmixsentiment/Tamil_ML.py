# # IMPORTS
# import pandas as pd
# import numpy as np
# import copy
# import re
# import nltk
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils import class_weight
# from scipy.sparse import csr_matrix

# # Fix: Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer, LancasterStemmer

# wordnet_lemmatizer = WordNetLemmatizer()
# lancaster_stemmer = LancasterStemmer()

# # LOAD DATA
# df = pd.read_csv('Tamil_first_ready_for_sentiment.csv', sep='\t', names=['category', 'text'])
# text = df[['text']]
# labels = df[['category']]

# # DATA CLEANING FUNCTIONS
# def take_data_to_shower(tweet):
#     noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
#     for noise in noises:
#         tweet = tweet.replace(noise, '')
#     return re.sub(r'[^a-zA-Z஀-௿]', ' ', tweet)  # Keep Tamil characters

# def tokenize(tweet):
#     return word_tokenize(tweet.lower())

# def remove_stop_words(tokens):
#     stop_words = set(stopwords.words('english'))
#     return [token for token in tokens if token not in stop_words and len(token) > 1]

# def stem_and_lem(tokens):
#     return [lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(token)) for token in tokens if len(token) > 1]

# # APPLY CLEANING PROCESS
# tqdm.pandas()
# clean_texts = copy.deepcopy(text)

# clean_texts['text'] = text['text'].progress_apply(take_data_to_shower)
# clean_texts['tokens'] = clean_texts['text'].progress_apply(tokenize)
# clean_texts['tokens'] = clean_texts['tokens'].progress_apply(remove_stop_words)
# clean_texts['tokens'] = clean_texts['tokens'].progress_apply(stem_and_lem)

# text_vector = clean_texts['tokens'].tolist()

# # TEXT VECTORIZATION (Using Sparse Matrices)
# def tfid(text_vector):
#     vectorizer = TfidfVectorizer(max_features=5000) 
#     untokenized_data = [' '.join(tweet) for tweet in tqdm(text_vector, desc="Vectorizing...")]
#     vectorizer = vectorizer.fit(untokenized_data)
#     vectors = vectorizer.transform(untokenized_data)
#     return vectorizer, csr_matrix(vectors)  
# vectorizer, vectors_a = tfid(text_vector)
# labels_a = np.array(labels['category'].values)  #  Convert labels to NumPy array

# # CLASSIFICATION MODELS
# def compute_class_weight_dictionary(y):
#     classes = np.unique(y)
#     class_weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
#     return dict(zip(classes, class_weights))

# def classify(vectors, labels, model_type="DT"):
#     train_vectors, test_vectors, train_labels, test_labels = train_test_split(
#         vectors, labels, random_state=5, test_size=0.2
#     )

#     class_weights = compute_class_weight_dictionary(train_labels)

#     classifiers = {
#         "MNB": MultinomialNB(alpha=0.7),  # Removed `class_weight`
#         "KNN": KNeighborsClassifier(n_jobs=-1),  # Removed `class_weight`
#         "SVM": SVC(class_weight=class_weights, kernel='linear'),
#         "DT": DecisionTreeClassifier(max_depth=100, min_samples_split=5, class_weight=class_weights),  # Fix: Reduce depth
#         "RF": RandomForestClassifier(max_depth=100, min_samples_split=5, class_weight=class_weights),
#         "LR": LogisticRegression(multi_class='auto', solver='lbfgs', class_weight=class_weights)
#     }

#     if model_type not in classifiers:
#         print(" Wrong Classifier Type!")
#         return

#     classifier = classifiers[model_type]

#     #  Optimize GridSearchCV with limited parameters
#     if model_type in ["KNN", "SVM", "DT", "RF", "LR"]:
#         params = {
#             "KNN": {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']},  # Reduced parameter range
#             "SVM": {'C': [0.1, 1]},
#             "DT": {'criterion': ['gini', 'entropy']},
#             "RF": {'n_estimators': [50, 100], 'criterion': ['gini', 'entropy']},
#             "LR": {"C": [0.1, 1], "penalty": ["l2"]}
#         }
#         classifier = GridSearchCV(classifier, params.get(model_type, {}), cv=3, n_jobs=1)  #  Set `n_jobs=1`
    
#     classifier.fit(train_vectors, train_labels)

#     # Fix: Only use `.best_estimator_` if GridSearch was used
#     if isinstance(classifier, GridSearchCV):
#         classifier = classifier.best_estimator_

#     train_accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
#     test_predictions = classifier.predict(test_vectors)
#     test_accuracy = accuracy_score(test_labels, test_predictions)

#     print(f"\n Model: {model_type}")
#     print(f" Training Accuracy: {train_accuracy:.4f}")
#     print(f" Test Accuracy: {test_accuracy:.4f}")
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(test_labels, test_predictions))
#     print("\nClassification Report:")
#     print(classification_report(test_labels, test_predictions))
#     return classifier

# # TRAIN MODELS
# model = classify(vectors_a, labels_a, "SVM")
# # model = classify(vectors_a, labels_a, "RF")  # Random Forest instead of SVM. same as this we can give the other model.


# # Function to analyze user input
# def analyze_comment(comment):
#     cleaned = take_data_to_shower(comment)
#     tokens = tokenize(cleaned)
#     tokens = remove_stop_words(tokens)
#     tokens = stem_and_lem(tokens)
#     vectorized_comment = vectorizer.transform([' '.join(tokens)])
#     prediction = model.predict(vectorized_comment)[0]
#     return prediction

# # Terminal Input for User Sentiment Analysis
# while True:
#     user_input = input("\nEnter a comment (or type 'exit' to stop): ")
#     if user_input.lower() == "exit":
#         break
#     sentiment = analyze_comment(user_input)
#     print(f"Sentiment: {sentiment}")








# # IMPORTS
# import pandas as pd
# import numpy as np
# import copy
# import re
# import nltk
# import uvicorn
# from tqdm import tqdm
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.keys import Keys
# from webdriver_manager.chrome import ChromeDriverManager
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.svm import SVC
# from sklearn.utils import class_weight
# from scipy.sparse import csr_matrix
# import time

# # NLTK Downloads
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer, LancasterStemmer

# # FastAPI App
# app = FastAPI()

# # Allow CORS for frontend interaction
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins (update in production)
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Text Processing Tools
# wordnet_lemmatizer = WordNetLemmatizer()
# lancaster_stemmer = LancasterStemmer()

# # Load Training Data
# df = pd.read_csv('Tamil_first_ready_for_sentiment.csv', sep='\t', names=['category', 'text'])
# text = df[['text']]
# labels = df[['category']]

# # Data Cleaning Functions
# def take_data_to_shower(tweet):
#     noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
#     for noise in noises:
#         tweet = tweet.replace(noise, '')
#     return re.sub(r'[^a-zA-Z஀-௿]', ' ', tweet)  # Keep Tamil characters

# def tokenize(tweet):
#     return word_tokenize(tweet.lower())

# def remove_stop_words(tokens):
#     stop_words = set(stopwords.words('english'))
#     return [token for token in tokens if token not in stop_words and len(token) > 1]

# def stem_and_lem(tokens):
#     return [lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(token)) for token in tokens if len(token) > 1]

# # Apply Cleaning Process
# tqdm.pandas()
# clean_texts = copy.deepcopy(text)
# clean_texts['text'] = text['text'].progress_apply(take_data_to_shower)
# clean_texts['tokens'] = clean_texts['text'].progress_apply(tokenize)
# clean_texts['tokens'] = clean_texts['tokens'].progress_apply(remove_stop_words)
# clean_texts['tokens'] = clean_texts['tokens'].progress_apply(stem_and_lem)

# text_vector = clean_texts['tokens'].tolist()

# # Text Vectorization
# def tfid(text_vector):
#     vectorizer = TfidfVectorizer(max_features=5000)
#     untokenized_data = [' '.join(tweet) for tweet in text_vector]
#     vectorizer = vectorizer.fit(untokenized_data)
#     vectors = vectorizer.transform(untokenized_data)
#     return vectorizer, csr_matrix(vectors)

# vectorizer, vectors_a = tfid(text_vector)
# labels_a = np.array(labels['category'].values)

# # Classification Model Training
# def classify(vectors, labels):
#     train_vectors, test_vectors, train_labels, test_labels = train_test_split(
#         vectors, labels, random_state=5, test_size=0.2
#     )
    
#     class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
#     class_weights = dict(zip(np.unique(train_labels), class_weights))

#     model = SVC(class_weight=class_weights, kernel='linear')
#     model.fit(train_vectors, train_labels)

#     test_predictions = model.predict(test_vectors)
#     test_accuracy = accuracy_score(test_labels, test_predictions)

#     return {
#         "model": "SVM",
#         "test_accuracy": test_accuracy,
#         "confusion_matrix": confusion_matrix(test_labels, test_predictions).tolist(),
#         "classification_report": classification_report(test_labels, test_predictions, output_dict=True)
#     }

# model_results = classify(vectors_a, labels_a)

# # Scrape YouTube Comments
# def scrape_youtube_comments(video_url):
#     options = webdriver.ChromeOptions()
#     options.add_argument("--headless=new")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")

#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     driver.get(video_url)

#     time.sleep(5)
#     for _ in range(5):  
#         driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
#         time.sleep(2)

#     comments = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
#     comment_list = [comment.text for comment in comments if comment.text.strip()]

#     driver.quit()
#     return comment_list

# # API Endpoint to Analyze YouTube Video
# class VideoURL(BaseModel):
#     url: str

# @app.post("/analyze-youtube/")
# async def analyze_youtube(video: VideoURL):
#     comments = scrape_youtube_comments(video.url)
#     if not comments:
#         raise HTTPException(status_code=404, detail="No comments found.")

#     return model_results

# # Run the app
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# IMPORTS
import pandas as pd
import numpy as np
import copy
import re
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from googleapiclient.discovery import build
import os

# Fix: Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from dotenv import load_dotenv
load_dotenv()  # ✅ Load .env variables early

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in environment. Please set it in your .env file.")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# TEXT PROCESSING
wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()


def take_data_to_shower(tweet):
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

# LOAD DATA
df = pd.read_csv('Tamil_first_ready_for_sentiment.csv', sep='\t', names=['category', 'text'])
text = df[['text']]
labels = df[['category']]

# CLEANING
tqdm.pandas()
clean_texts = copy.deepcopy(text)
clean_texts['text'] = text['text'].progress_apply(take_data_to_shower)
clean_texts['tokens'] = clean_texts['text'].progress_apply(tokenize)
clean_texts['tokens'] = clean_texts['tokens'].progress_apply(remove_stop_words)
clean_texts['tokens'] = clean_texts['tokens'].progress_apply(stem_and_lem)

text_vector = clean_texts['tokens'].tolist()

# VECTORIZE
def tfid(text_vector):
    vectorizer = TfidfVectorizer(max_features=5000) 
    untokenized_data = [' '.join(tweet) for tweet in tqdm(text_vector, desc="Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data)
    return vectorizer, csr_matrix(vectors)  

vectorizer, vectors_a = tfid(text_vector)
labels_a = np.array(labels['category'].values)  

# MODEL
def classify(vectors, labels):
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(
        vectors, labels, random_state=5, test_size=0.2
    )

    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

    classifier = SVC(class_weight=class_weight_dict, kernel='linear')
    classifier.fit(train_vectors, train_labels)

    test_predictions = classifier.predict(test_vectors)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f"\n Model: SVM")
    print(f" Test Accuracy: {test_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    
    return classifier

model = classify(vectors_a, labels_a)

# ANALYSIS FUNCTION
def analyze_comment(comment):
    cleaned = take_data_to_shower(comment)
    tokens = tokenize(cleaned)
    tokens = remove_stop_words(tokens)
    tokens = stem_and_lem(tokens)
    vectorized_comment = vectorizer.transform([' '.join(tokens)])
    prediction = model.predict(vectorized_comment)[0]
    return prediction

# FUNCTION TO SCRAPE YOUTUBE COMMENTS
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

# DYNAMIC ANALYSIS
def analyze_youtube_comments_runtime():
    video_url = input("Enter YouTube video URL: ")
    comments = get_youtube_comments(video_url)
    if not comments:
        print("No comments found or failed to fetch.")
        return
    
    for comment in comments:
        sentiment = analyze_comment(comment)
        print(f"Comment: {comment}\nSentiment: {sentiment}\n")

# TERMINAL UI
while True:
    choice = input("\nEnter a comment to analyze, 'youtube' to analyze YouTube video comments, or 'exit' to quit: ")
    if choice.lower() == "exit":
        break
    elif choice.lower() == "youtube":
        analyze_youtube_comments_runtime()
    else:
        sentiment = analyze_comment(choice)
        print(f"Sentiment: {sentiment}")
