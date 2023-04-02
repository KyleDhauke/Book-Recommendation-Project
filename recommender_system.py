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


#print("before duplicates removed: " + str(df.shape)) # (5976479, 8)
df1= df.drop_duplicates(['original_title'])
#print("after original_title duplicates removed: " + str(df1.shape)) #(5859358, 8)
# 117,121 duplicates removed


### Matrix Factorisation ###

books_matrix = df1.pivot_table(index = 'user_id', columns = 'original_title', values = 'rating', fill_value = 0.0)
# print(books_matrix.shape) #(3821, 9274)
# print(books_matrix.head()) #[5 rows x 9274 columns]



# Creating a training data set
X = books_matrix.values.T # (9274, 3821). Transposed the books_matrix.

#Fitting the Model

n_comp = 12 # Variable to decide the n_components used in our truncated SVD.
SVD = TruncatedSVD(n_components=n_comp, random_state=0)
matrix = SVD.fit_transform(X)

var_explained = SVD.explained_variance_ratio_.sum() # Stores the percentage of the variance between eigenvalues

print(matrix.shape) #(9274, 12) for n_components = 12
print(var_explained * 100) #6.839 for n_components = 12



# RESEARCH: UP TO HERE SO FAR


# import warnings
# warnings.filterwarnings("ignore",category =RuntimeWarning)#to avoid RuntimeWarning #Base class for warnings about dubious runtime behavior.
# corr = np.corrcoef(matrix)
# corr.shape
#
#
#
# title = books_matrix.columns
# title_list = list(title)
# samia = title_list.index('Memoirs of a Geisha')
# corr_samia  = corr[samia]
# print(list(title[(corr_samia >= 0.9)]))






# Close read files later!!
