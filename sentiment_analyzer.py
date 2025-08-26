import nltk
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#  Download NLTK movie reviews (first run only)
nltk.download("movie_reviews")
from nltk.corpus import movie_reviews

#  Load reviews(words, labels)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)  # mix order

#  Convert words → sentence + clean text
reviews = [" ".join(words) for words, label in documents]
labels = ["positive" if label == "pos" else "negative" for words, label in documents]

#  Put into DataFrame
df = pd.DataFrame({"review": reviews, "sentiment": labels})
print(df.head())   # Showing first 5 rows

#  train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42)

#  TF-IDF featrures
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#  Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

#  Accuracy
accuracy = model.score(X_test_vectorized, y_test)
print("Model Accuracy:", accuracy)

#  Try user reviews
while True:
    user_input = input("Enter a review (or type 'quit' to stop): ")
    if user_input.lower() == "quit":
        break
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)
    print("Sentiment →", prediction[0])
