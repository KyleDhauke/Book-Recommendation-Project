import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
import warnings
from scipy.sparse import linalg
from sklearn.utils.extmath import randomized_svd

# Function which uses basic matrix factorisation to recommended n_recommendations amount of books. Optionally, you can choose
# the amount of eigenvalues to keep in our truncatedSVD. The higher the value, the more biased (personalised).


def matrix_factorisation_2(book_title, n_recommendations = 10, n_comp = 12):
    """
        Uses advanced matrix factorisiation to recommend n_recommendations amount of books.
        Introduces L2 Regularisation and biases.

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
    X = books_matrix.values.T

    # Mean and standard deviation of each column
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Applying z-score normalization to reduce the CPU load.
    X_normalized = (X - mean) / std

    # Calculate the truncated SVD with regularization
    U, Sigma, Vt = randomized_svd(X_normalized, n_components=n_comp)
    Sigma_reg = np.diag(Sigma)  # Diagonal matrix of singular values

    # Apply L2 regularization to the singular values
    lambda_reg = 0.01  # Regularization parameter
    Sigma_reg = np.sqrt(Sigma_reg ** 2 + lambda_reg)

    # Computes the factor matrices with regularization
    matrix = U.dot(Sigma_reg)
    Vt = Sigma_reg.dot(Vt)


    # Initialize biases
    user_biases = np.zeros(X_normalized.shape[0])
    item_biases = np.zeros(X_normalized.shape[1])
    
    # Fit the regularized model
    n_iterations = 12  # Number of iterations for optimization
    learning_rate = 0.001  # Learning rate for optimization

    for _ in range(n_iterations):
        for i in range(X_normalized.shape[0]):
            for j in range(X_normalized.shape[1]):
                if X_normalized[i, j] > 0:
                    prediction = np.dot(matrix[i, :], Vt[:, j]) + mean[j] + user_biases[i] + item_biases[j]
                    error = X_normalized[i, j] - prediction

                    # Update factor matrices
                    matrix[i, :] += learning_rate * (error * Vt[:, j] - lambda_reg * matrix[i, :])
                    Vt[:, j] += learning_rate * (error * matrix[i, :] - lambda_reg * Vt[:, j])

                    # Update biases
                    user_biases[i] += learning_rate * (error - lambda_reg * user_biases[i])
                    item_biases[j] += learning_rate * (error - lambda_reg * item_biases[j])

    import warnings
    warnings.filterwarnings("ignore",category =RuntimeWarning) #avoids RuntimeWarning #Base class for warnings about dubious runtime behavior.
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


   





