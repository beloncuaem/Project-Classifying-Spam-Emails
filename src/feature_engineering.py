# src/feature_engineering.py
# Người thực hiện: Nhật Trung
# Chức năng: Tạo features từ email để train model

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def build_tfidf_features(texts, max_features=10000, ngram_range=(1, 2)):
    """
    Tạo TF-IDF features từ text email
    Input:
        - texts: list các email text
        - max_features: số lượng từ tối đa (mặc định 10000)
        - ngram_range: (1,2) là unigram + bigram
    Output:
        - tfidf_matrix: ma trận TF-IDF dạng sparse
        - vectorizer: object đã fit để dùng sau
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def extract_manual_features(texts):
    """
    Tạo feature thủ công từ email
    Input: list các email text
    Output: DataFrame các feature:
        - length: độ dài email
        - url_count: số URL
        - exclamation_count: số dấu !
        - dollar_count: số ký tự $
        - uppercase_ratio: tỷ lệ chữ in hoa
    """
    features = []

    for text in texts:
        # Độ dài email
        length = len(text)

        # Số URL (http, https, www)
        url_count = len(re.findall(r'https?://|www\.', text))

        # Số dấu !
        exclamation_count = text.count('!')

        # Số ký tự $
        dollar_count = text.count('$')

        # Tỷ lệ chữ in hoa (tránh chia cho 0)
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / (len(text) + 1)

        features.append([length, url_count, exclamation_count, dollar_count, uppercase_ratio])

    return pd.DataFrame(features,
                        columns=['length', 'url_count', 'exclamation_count', 'dollar_count', 'uppercase_ratio'])


def combine_features(tfidf_matrix, manual_df):
    """
    Ghép TF-IDF và manual features
    Input:
        - tfidf_matrix: ma trận TF-IDF (sparse)
        - manual_df: DataFrame manual features
    Output:
        - combined: ma trận đã ghép (sparse)
    """
    manual_sparse = sp.csr_matrix(manual_df.values)
    combined = sp.hstack([tfidf_matrix, manual_sparse])
    return combined