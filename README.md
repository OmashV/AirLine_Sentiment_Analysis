# Airline Sentiment Analysis

This project performs sentiment analysis on airline tweets using machine learning techniques. The dataset consists of tweets labeled with sentiments such as "positive," "neutral," and "negative." The project involves data preprocessing, feature extraction, model training, and evaluation.

## Dataset
The dataset used is `Tweets.csv`, which contains 14,640 rows and 15 columns. We have focused on two columns:
- `airline_sentiment`: Sentiment of the tweet (positive, neutral, negative).
- `text`: The actual tweet content.

## Installation

Ensure you have Python installed and then install the required dependencies by running:

```bash
pip install pandas nltk scikit-learn
```

## Steps

### 1. Load the Dataset

```python
import pandas as pd

df = pd.read_csv('Tweets.csv')
df = df[["airline_sentiment", "text"]]
print(f"Shape of the dataset: {df.shape}")
```

### 2. Preprocessing

Using NLTK for text preprocessing, including:
- Converting text to lowercase
- Removing URLs
- Tokenization
- Removing stopwords
- Stemming

```python
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http.?://[\S]+[\s]?', '', text)
    text = nltk.word_tokenize(text)
    text = [ps.stem(i) for i in text if i not in stopwords.words('english')]
    return " ".join(text)

df["text_cleaned"] = df["text"].apply(clean_text)
```

### 3. Exploratory Data Analysis

```python
print(df.groupby("airline_sentiment").size())
neutral_tweets = df[df["airline_sentiment"] == "neutral"]
print(neutral_tweets["text"].nunique())
```

### 4. Feature Extraction

We use TF-IDF vectorization to convert the cleaned text data into numerical format.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)
x = tfidf.fit_transform(df["text_cleaned"]).toarray()
y = df['airline_sentiment'].values
```

### 5. Model Training and Evaluation

Two models are trained:
- RandomForestClassifier
- MultinomialNB (Naive Bayes)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Random Forest Model
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Naive Bayes Model
model_nb = MultinomialNB()
model_nb.fit(x_train, y_train)
y_pred_nb = model_nb.predict(x_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
```

### 6. Results

The model accuracies obtained:
- Random Forest: `75.03%`
- Naive Bayes: `72.19%`

## Conclusion

- The Random Forest model performed better than Naive Bayes.
- Further improvements can be made by fine-tuning hyperparameters and exploring other models.

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd airline-sentiment-analysis
   ```
3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

## License
This project is licensed under the MIT License.

