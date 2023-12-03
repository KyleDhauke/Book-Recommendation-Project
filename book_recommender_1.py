import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
import warnings
import time

# Function which uses basic matrix factorisation to recommended n_recommendations amount of books. Optionally, you can choose
# the amount of eigenvalues to keep in our truncatedSVD. The higher the value, the more biased (personalised).


def matrix_factorisation_1(book_title, n_recommendations = 10, n_comp = 12):
    """
        Uses basic matrix factorisiation to recommend n_recommendations amount of books.

        Parameters:
            book_title (string): A string which contains a book title from the Goodbooks-10k Dataset.
            n_recommendations (int): Integer which is the amount of recommendations that are returned. Default is 10.
            n_comp (int): Integer value denoting the amount of eigenvalues to keep in our truncatedSVD. The higher the value, the more
            biased the recommendations (personalised). Default is 12.
        
            Returns:
            recommended_books (list): A list of strings containing an n_recommendations amount of books that were recommended.
    """
    books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep=',')
    books = books.iloc[:, :16] # Splices the first 16 columns.
    books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'original_publication_year','language_code','work_ratings_count','work_text_reviews_count'])

    ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')
    df = pd.merge(ratings, books, on="book_id")


    #Before duplicates were removed: (5976479, 8)
    df1= df.drop_duplicates(['original_title'])
    #After duplicates were removed: (5859358, 8)
    # 117,121 duplicates removed


    ### Matrix Factorisation ###
    books_matrix = df1.pivot_table(index = 'user_id', columns = 'original_title', values = 'rating', fill_value = 0.0)

    # Creating a training data set
    X = books_matrix.values.T # (9274, 3821). Transposes the books_matrix.

    #Fitting the Model
    SVD = TruncatedSVD(n_components=n_comp, random_state=0) # Variable to decide the n_components used in our truncated SVD.
    matrix = SVD.fit_transform(X)

    var_explained = SVD.explained_variance_ratio_.sum() # Stores the percentage of the variance between eigenvalues. Used in evaluation.
    # print(matrix.shape) #(9274, 12) for n_components = 12
    print("the current var= ", round(var_explained * 100,3)) #Percentage of variance. 6.839% for n_components = 12


    import warnings
    warnings.filterwarnings("ignore",category =RuntimeWarning)#avoids RuntimeWarning #Base class for warnings about dubious runtime behavior.
    corr = np.corrcoef(matrix)
    corr.shape


    title = books_matrix.columns
    title_list = list(title)
    samia = title_list.index(book_title)
    corr_samia  = corr[samia]

    book_corr_list = list(zip(title_list,corr_samia)) # Zips the book correlation coefficients to their respective books
    sorted_book_corr = sorted(book_corr_list, key = lambda x: x[1],reverse = True) # Sorts the list from descending order

    recommended_books  = [] # Creates an empty list to store the recommended books.
    for i in sorted_book_corr:
        x = i[0]
        if x != book_title:
            recommended_books.append(x)

    return (recommended_books[:n_recommendations]) # Returns the n_recommendations amount of books.



def get_running_time(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

# running_time = round(get_running_time(matrix_factorisation_1, "Catching Fire", 10, 3821),3)
# print(f"Total running time: {running_time} seconds")
