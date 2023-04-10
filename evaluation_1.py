from recommender_system import matrix_factorisation_1
import numpy as np

book_title = 'Memoirs of a Geisha'
n_recommendations = 20

print(matrix_factorisation_1(book_title,n_recommendations))


# books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep=',')
# books = books.iloc[:, :16] # Splices the first 16 columns.
# books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'original_publication_year','language_code','work_ratings_count','work_text_reviews_count'])
#
#
# ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')
