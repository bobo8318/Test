from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

news = fetch_20newsgroups(subset='all')
#tfdivect = TfidfVectorizer.fit_transform()
print(len(news.data))
print(news.target[0])
