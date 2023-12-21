import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import re
import string
from nltk.stem import PorterStemmer

def formatting_data(data):
    all_ratings = []

    for name, ratings in tqdm(zip(data['name'], data['reviews_list'])):
        ratings = eval(ratings)

        for score, doc in ratings:
            if score:
                score = score.strip("Rated").strip()
                doc = doc.strip('RATED').strip()
                score = float(score)

                all_ratings.append([name, score, doc])

    result_df = pd.DataFrame(all_ratings, columns=['name', 'rating', 'review'])

    result_df['name'] = result_df['name'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', "", x))
    result_df['review'] = result_df['review'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', "", x))

    return result_df

def duplicate_reviews(data, min_length, min_count):
    review_counts_df = data['review'].value_counts().reset_index()
    review_counts_df.columns = ['review', 'count']

    mask = (review_counts_df['review'].str.len() >= min_length) & (review_counts_df['count'] >= min_count)

    rating_df = data[~data['review'].isin(review_counts_df[mask]['review'])]

    return rating_df

def clean_text(text, rating):
    if pd.isna(text):
        return 'good' if rating >= 3 else 'bad'

    words = text.lower().split()

    words = [word.lower().translate(str.maketrans('', '', string.punctuation)) for word in words]

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    processed_text = ' '.join(words)

    return processed_text

def map_to_sentiment(rating):
    if rating >= 3:
        return 'positive'  # If true, return 'positive'

    else:
        return 'negative'  # If true, return 'negative'

def convert_to_label(sentiment):
    if sentiment == 'positive':
        return 1  # If true, return the label 1 (or any value representing positive)

    elif sentiment == 'negative':
        return 0  # If true, return the label 0 (or any value representing negative)

def main_processing_data(data):
    result_df = formatting_data(data)
    print('Formatting data sucessfully!!!')
    result_df = duplicate_reviews(result_df, 14, 2)
    print('Remove dulicate data sucessfully!!!')
    result_df['review'] = result_df.apply(lambda row: clean_text(row['review'], row['rating']), axis=1)
    print('\nClean data sucessfully!!!')
    result_df['sentiment'] = result_df['rating'].apply(map_to_sentiment)
    print('\nAdd sentiment column sucessfully!!!')
    result_df['label'] = result_df['sentiment'].apply(convert_to_label)
    print('\nAdd binary label column sucessfully!!!')
    result_df = result_df[['name', 'review','rating','sentiment','label']]
    result_df = result_df[result_df['review']!='']
    print('\nDrop NaN rows sucessfully!!!')
    return result_df

path = './zomato.csv'
df = pd.read_csv(path)
file_path = 'Ratings.csv'
main_processing_data(df).to_csv(file_path, index=False)
print('\nLưu file csv sau khi thực hiện xử lý thành công!!!')