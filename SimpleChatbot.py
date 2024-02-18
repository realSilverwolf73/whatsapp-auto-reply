import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

class ChatbotClassifier:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.chatbot_corpus = self.load_corpus()

    def load_corpus(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as file:
            corpus_data = json.load(file)
        return corpus_data['chatbot_corpus']

    def preprocess_text(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def train_classifier(self):
        questionS = [self.preprocess_text(entry['question']) for entry in self.chatbot_corpus]
        intentS = [entry['intent'] for entry in self.chatbot_corpus]

        X_train, X_test, y_train, y_test = train_test_split(questionS, intentS, test_size=0.2, random_state=42)

        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vectorized, y_train)

        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vectorized)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    def predict_intent(self, user_query):
        preprocessed_query = self.preprocess_text(user_query)
        query_vectorized = self.vectorizer.transform([preprocessed_query])
        predicted_intent = self.classifier.predict(query_vectorized)

        for entry in self.chatbot_corpus:
            if entry['intent'] == predicted_intent[0]:
                return entry['response']
