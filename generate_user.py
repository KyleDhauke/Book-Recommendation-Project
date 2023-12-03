import pandas as pd
import numpy as np
import random
from math import isnan


to_read_with_nan = pd.read_csv('Goodbooks-10k Dataset/to_read.csv', sep = ',')
to_read = to_read_with_nan.dropna()
max_users = to_read.max()[0] # 53424 users, however, in the to_read dataset only 48871 users rated books.




def _check_user(x = 0): # Set default to 0 to avoid argument confusion.
    """ Recursively picks a random user_id (int) and checks if they want to read at least 10 books from the 
        Goodbooks-10k to_read dataset.
        
        Returns chosen_user (int), an id of a user.
    """
    random_user = random.randint(0, max_users) #Generates a random number, capped at the maximum users.
    duplicate_count = to_read.pivot_table(index = ['user_id'], aggfunc ='size') #Creates a pivot table around users who have read multiple books.
    # On average, most users want to_read at least 18 books.
    chosen_user = x #Recursively stores the chosen_user

    if random_user in duplicate_count.index:
        if duplicate_count[random_user] <= 9:
            return _check_user(random_user) # Fail: If user has not read more than 10 books, repeat.
        if duplicate_count[random_user] >= 10:
            chosen_user = random_user # Success: If user has read more than 10 books
    else:
        return _check_user(random_user) # Fail: If the user is not in the index, repeat.

    return chosen_user


def pick_test_user():
    """
    Retrieves a suitable test_user id and returns data neccessary to test the recommendation system.

    Returns:
        test_user (int): id of a chosen test user.
        test_book_title (string): A string containing the title one of the test_user highest rated books.
        test_book_id (id): An integer id of one of the books which the test_user has rated the highest.
        test_book_rating (id): An integer value between 1-5 denoting the rating of the test book. Typically it is 5.
    """
    books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep=',')
    books = books.iloc[:, :16] # Splices the first 16 columns.
    books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'original_publication_year','language_code','work_ratings_count','work_text_reviews_count'])
    books = books.dropna(subset=['original_title'])
    ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')

    test_user = _check_user() # Picks a random user
    a = ratings.loc[ratings['user_id'] == test_user] # Isolates all the books the test_user has rated.
    a = a.sort_values(by=['rating'], ascending=False)
    list_a = list(a.iloc[0]) # Stores the highest rated books into a list.
    test_book_id = list_a[1]

    book_info = books.loc[books['book_id'] == test_book_id] # Stores the information of the tested book.

    try:
        test_book_title = book_info.values[0][3] # Stores the title of the tested book.
        if pd.isna(test_book_title):
            raise IndexError("Book title is out of bounds")  # Raise an exception if the book title is NaN or out of bounds
    except (IndexError, KeyError) as e:
        print("Exception:", str(e))
        return pick_test_user()  # Recursive call to pick a new test user

    test_book_rating = list_a[2]

    return test_user, test_book_title, test_book_id, test_book_rating
