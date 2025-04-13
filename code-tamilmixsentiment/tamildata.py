# import re
# import nltk 
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# lemmetizer = WordNetLemmatizer()
# nltk.download('stopwords')

# #understanding the data
# import pandas as pd 
# data = pd.read_csv('thiru.csv',sep ='\t', names=['category', 'text'])#getting the input fro the thiru.csv

# # print (data.head(3))# to print the top 3 data
# print(data.isnull().sum())#if is null prints the sum of null

# data1 = data.dropna()#removes null

# # print (data1.head(3))
# print(data1.isnull().sum())

# print(data1['category'].value_counts())# to print how may 0,1,2,3,4 cases.
# # print (data.head(10))

# duplicate = data1.duplicated()# to get the duplicated
# print({duplicate.sum()})#to print the sum of duplicated
# data1 = data1.drop_duplicates()#  to remove the duplicates

# duplicate = data1.duplicated()
# print({duplicate.sum()})
# print(data1.isnull().sum())
# print(data1.shape)# to count the total lines.


# print(data1)

# # to change all to lowercase
# data1 = data1.map(lambda x:x.lower() if isinstance(x,str) else x )
# # print(data1.head(5))


# #remove punctuation and remove special characters
# data1 = data1.map(lambda x: re.sub(r'[^a-zA-Z\s]', '',str(x)) if isinstance(x,str) else x )
# # print(data1.head(10))

# #stopwords removing

# stop_words = set(stopwords.words('english'))
# data1['text'] = data1['text'].split()
# def remove_stop_words(text):
#     return ' '.join ([word for word in text.split() if word.lower() not in stop_words])
#     tokendata['text_coloum'] = tokendata['text_coloumn'].apply(remove_stopwords)

# tokendata = [lemmatizer.lemmatize(word) for word in tokendata if word not in stop_words]
# tokendata = ' '.join(tokendata) 

# print(tokendata.head(10))



###############################################################################

# import re
# import nltk
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Download stopwords
# nltk.download('stopwords')

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load dataset
# data = pd.read_csv('thiru.csv', sep='\t', names=['category', 'text'])

# # Check if 'text' column exists
# if 'text' not in data.columns:
#     raise ValueError("The 'text' column is missing in the dataset!")

# # Drop rows with missing values
# data1 = data.dropna().copy()  # Using .copy() to avoid warnings

# # Convert all text to lowercase
# data1['cleaned_text'] = data1['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)

# # Remove punctuation and special characters
# data1['cleaned_text'] = data1['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)

# # Stopword removal
# stop_words = set(stopwords.words('english'))
# def remove_stop_words(text):
#     return ' '.join([word for word in text.split() if word not in stop_words])

# data1['cleaned_text'] = data1['cleaned_text'].apply(remove_stop_words)

# # Lemmatization
# data1['cleaned_text'] = data1['cleaned_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# # Check if 'cleaned_text' column exists before using it
# if 'cleaned_text' not in data1.columns:
#     raise ValueError("The 'cleaned_text' column was not created successfully!")

# # Initialize TF-IDF Vectorizer
# vectorizer = TfidfVectorizer()

# # Fit and transform cleaned text
# X = vectorizer.fit_transform(data1['cleaned_text']).toarray()

# # Print first 5 TF-IDF transformed values
# print(X[:5])

#############################################################################

import random
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load dataset
data = pd.read_csv('thiru.csv', sep='\t', names=['category', 'text'])

# Check for null values
print("Null values before cleanup:\n", data.isnull().sum())

# Drop rows with missing values
data1 = data.dropna().copy()  # Using .copy() to avoid modifying the original DataFrame
print("Null values after cleanup:\n", data1.isnull().sum())

# Check for duplicate rows
print("Duplicate count before removal:", data1.duplicated().sum())

# Remove duplicate rows
data1 = data1.drop_duplicates()
print("Duplicate count after removal:", data1.duplicated().sum())

# Check shape after cleaning
print("Dataset shape after cleaning:", data1.shape)

# Create a new column for cleaned text
data1['cleaned_text'] = data1['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)

# Remove punctuation and special characters
data1['cleaned_text'] = data1['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)

# Stopword removal function
stop_words = set(stopwords.words('english'))
def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# Apply stopword removal
data1['cleaned_text'] = data1['cleaned_text'].apply(remove_stop_words)

# Lemmatization
data1['cleaned_text'] = data1['cleaned_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Print first 10 rows after preprocessing
print(data1.head(10))

# ✅ Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data1['cleaned_text']).toarray()

# Print the shape of the TF-IDF matrix
print("TF-IDF matrix shape:", X.shape)
print(X)

A = data1['text_column']
Y = data1['label_column']

A_train, A_test, Y_train, Y_test = train_test_split(A, Y, test_size = 0.2, random_state=42)


print("A_train data :")
print(A_train.head())

print("A test data :")
print(A_test.head())

x_train_tfidf = vectorizer.transform(A_train).toarray()
x_test_tfidf = vectorizer.transform(A_test).toarray()

classifier = LogisticRegression()
classifier.fit(x_train_tfidf, Y_train)

Y_pred = classifier.predict(x_test_tfdif)

accurancy = accurancy_score(Y_test, Y_pred)
print(f'Accurancy : {accurancy * 100:.2f}%')

print('\nCLassification Report :')
print(classification_report(Y_test, Y_pred))