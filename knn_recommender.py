import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def knn_popularity_recommender(book_title, n_recommendations=10):
    # Read books and ratings datasets
    books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep=',')
    books = books.iloc[:, :16]
    books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13','original_publication_year', 'language_code', 'work_ratings_count','work_text_reviews_count'])
    books = books.dropna()
    ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')

   
    df1 = pd.merge(ratings, books, on="book_id")

    df= df1.drop_duplicates(['original_title'])  # Merge books and ratings datasets


    # Get the original titles of the books
    original_titles = books['original_title'].values
    original_titles = [title for title in original_titles if isinstance(title, str)]


    vectorizer = TfidfVectorizer(input='content')     # Initializes a TF-IDF vectorizer!


    tfidf_matrix = vectorizer.fit_transform(original_titles)     # Compute the TF-IDF vectors for the original book titles

  
    cosine_similarity_matrix = cosine_similarity(tfidf_matrix)   # Calculate the cosine similarity matrix


    title = books['original_title'].values
    title_list = list(title)
    samia = title_list.index(book_title)
    
    corr_samia = cosine_similarity_matrix[samia]

    book_corr_list = list(zip(title_list,corr_samia)) # Zips the book correlation coefficients to their respective books
    sorted_book_corr = sorted(book_corr_list, key = lambda x: x[1],reverse = True) # Sorts the list from descending order

    recommended_books  = [] # Creates an empty list to store the recommended books.
    for i in sorted_book_corr:
        x = i[0]
        if x != book_title:
            recommended_books.append(x)

    return (recommended_books[:n_recommendations])








