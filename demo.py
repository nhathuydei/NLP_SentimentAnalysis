import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Download stopwords if not already downloaded
nltk.download('stopwords')

class SentimentAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model_maxent_filename = './model/model_maxent.pickle'
        with open(self.model_maxent_filename, 'rb') as model_file:
            self.model_maxent = pickle.load(model_file)

        self.model_naivebayes_filename = './model/model_naivebayes.pickle'
        with open(self.model_naivebayes_filename, 'rb') as model_file:
            self.model_naivebayes = pickle.load(model_file)

        self.model_lstm_filename = './model/model_lstm.h5'
        self.model_lstm = load_model(self.model_lstm_filename)

        self.stop_words = set(stopwords.words('english'))  # English stopwords

        # Tokenizer for LSTM
        self.tokenizer = Tokenizer()
        self.maxlen = 95  # Adjust to the correct sequence length used during training

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel('Input your review:')
        layout.addWidget(self.label)

        self.text_input = QLineEdit(self)
        layout.addWidget(self.text_input)

        self.predict_button = QPushButton('Predict', self)
        self.predict_button.clicked.connect(self.predict_sentiment)
        layout.addWidget(self.predict_button)

        self.result_label_maxent = QLabel('')
        layout.addWidget(self.result_label_maxent)

        self.result_label_naivebayes = QLabel('')
        layout.addWidget(self.result_label_naivebayes)

        self.result_label_lstm = QLabel('')
        layout.addWidget(self.result_label_lstm)

        self.setLayout(layout)

    def predict_sentiment(self):
        input_review = self.text_input.text()
        words = nltk.word_tokenize(input_review)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in self.stop_words]
        features = dict([(word, True) for word in words])

        # Maxent prediction
        prediction_maxent = self.model_maxent.classify(features)
        sentiment_result_maxent = "Negative" if prediction_maxent == 0 else "Positive"
        keywords_maxent = ' '.join(words)
        result_text_maxent = (
            '\n' + "*" * 40 +
            f'\nMaxEnt Sentiment: \n{sentiment_result_maxent}\n' +
            "*" * 40
        )
        self.result_label_maxent.setText(result_text_maxent)

        # Naive Bayes prediction
        prediction_naivebayes = self.model_naivebayes.classify(features)
        sentiment_result_naivebayes = "Negative" if prediction_naivebayes == 0 else "Positive"
        keywords_naivebayes = ' '.join(words)
        result_text_naivebayes = (
            '\n' + "*" * 40 +
            f'\nNaive Bayes Sentiment: \n{sentiment_result_naivebayes}\n' +
            "*" * 40
        )
        self.result_label_naivebayes.setText(result_text_naivebayes)

        # Naive Bayes prediction
        prediction_naivebayes = self.model_naivebayes.classify(features)
        sentiment_result_naivebayes = "Negative" if prediction_naivebayes == 0 else "Positive"
        keywords_naivebayes = ' '.join(words)
        result_text_naivebayes = (
            '\n' + "*" * 40 +
            f'\nNaive Bayes Sentiment: \n{sentiment_result_naivebayes}\n' +
            "*" * 40
        )
        self.result_label_naivebayes.setText(result_text_naivebayes)

        # LSTM prediction
        self.tokenizer.fit_on_texts([input_review])
        sequence = self.tokenizer.texts_to_sequences([words])
        padded_sequence = pad_sequences(sequence, maxlen=self.maxlen)  # Adjusted to the correct sequence length
        prediction_lstm = self.model_lstm.predict(np.array(padded_sequence))
        sentiment_result_lstm = "Negative" if prediction_lstm < 0.5 else "Positive"
        keywords_lstm = ' '.join(words)
        result_text_lstm = (
            '\n' + "*" * 40 +
            f'\nLSTM Sentiment: \n{sentiment_result_lstm}\n' +
            "*" * 40 +
            f'\nKeywords: {keywords_lstm}\n'
        )
        self.result_label_lstm.setText(result_text_lstm)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SentimentAnalysisApp()
    window.setWindowTitle('Sentiment Analysis App')
    window.setGeometry(100, 100, 400, 300)
    window.show()
    sys.exit(app.exec())