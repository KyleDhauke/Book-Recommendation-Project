import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
import warnings



books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep=',')
books = books.iloc[:, :16] # Splices the first 16 columns.
books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'original_publication_year','language_code','work_ratings_count','work_text_reviews_count'])
# books.head(5)


ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')
# ratings.head(5)
df = pd.merge(ratings, books, on="book_id")
# df.head(5)

print("before duplicates removed: " + str(df.shape))

df1= df.drop_duplicates(['user_id','original_title'])
print("after original_title duplicates removed: " + str(df1.shape)) #(??, ??)

# print(df1.head(10)) #went down from ?? to ??






# Close read files later!!
