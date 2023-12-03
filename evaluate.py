import pandas as pd
import numpy as np
from generate_user import *
from book_recommender_1 import *
from book_recommender_2 import *
from knn_recommender import *

books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep = ',')
books = books.iloc[:, :16] # Splices the first 16 columns.
books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'original_publication_year','language_code','work_ratings_count','work_text_reviews_count'])
books = books.drop_duplicates(['original_title'])

to_read = pd.read_csv('Goodbooks-10k Dataset/to_read.csv', sep = ',')

ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep = ',')



def performance_metrics(n_tests,test_type, n_recommendations = 10, n_comp = 12):
    """
    Picks a recommendation algorithm to test and returns performance metrics based on a calculated confusion matrix.

    Parameters:
        n_tests (int): The number of tests to be ran on the chosen algorithm.
        test_type (string): The type of recommendation algorithm that will be tested. Pick between [b1_test,] 
        n_recommendations (int): The number of recommendations the algorithm will return. Default is 10.
        n_comp (int): Integer value denoting the amount of eigenvalues to keep when using an SVD method. Default is 12.
    
    Returns:
        accuracy [float]: A performance metric used to assess the accuracy of the recommendation algorithm.
        precision [float]: A performance metric used to assess the precision of the recommendation algorithm.
        recall [float]: A performance metric used to tell how many of the actual positive cases the algorithm could predict.
        f1_score [float]: A value denoting the harmonic mean of Precision and Recall.

    """

    performance_results = _confusion_matrix(n_tests, test_type, n_comp,n_recommendations)

    tp_total = performance_results[0]
    fp_total = performance_results[1]
    fn_total = performance_results[2]
    tn_total = performance_results[3]

    try:
        accuracy = (tp_total + tn_total)/((tp_total + fp_total + fn_total + tn_total))
    except ZeroDivisionError: #Deal with the error if we end up dividing with 0's.
        accuracy = 0

    try:
        precision = (tp_total) / (tp_total + fp_total)
    except ZeroDivisionError: #Deal with the error if we end up dividing with 0's.
        precision = 0

    try:
        recall = (tp_total) / (tp_total + fn_total)
    except ZeroDivisionError: #Deal with the error if we end up dividing with 0's.
        recall = 0

    try:
        f1_score = (2 / ((1/recall) + (1/precision)))
    except ZeroDivisionError: #Deal with the error if we end up dividing with 0's.
        f1_score = 0

    return accuracy, precision, recall, f1_score




def _confusion_matrix(n_tests, test_type, n_comp = 12, n_recommendations = 10): 
    """Takes in the selected test type and runs it n_tests amount of times, making sure the same user isn't picked more than once.
        Returns four confusion matrix values used to determine values in performance_metrics().
    """
    valid_test = ["b1_test","b2_test","knn_test"]

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    if n_tests < 1:
        raise ValueError("Minimum of 1 test required.")
    if test_type not in valid_test:
        raise ValueError("Test type must be one of %r." % valid_test)
    
    list_of_users = [] # Stores the list of users to be tested.
    i = 0
    while i < n_tests: #Creates x amount of unique users based on the amount of tests neccessary.
        curr_user = _check_user_list(list_of_users)
        list_of_users.append(curr_user)
        i += 1

    if test_type == valid_test[0]: # Determines the recommendation algorithm to be tested.
        for j in list_of_users: # For each user in the list of users, conduct a test and sum up their confusion matrix values.
            test_results = _b1_test(j[0],j[1],n_comp,n_recommendations,j[2])
            tp_total += test_results[0]
            fp_total += test_results[1]
            fn_total += test_results[2]
            tn_total += test_results[3]

    elif test_type == valid_test[1]: # Determines the recommendation algorithm to be tested.
        for j in list_of_users: # For each user in the list of users, conduct a test and sum up their confusion matrix values.
            test_results = _b2_test(j[0],j[1],n_comp,n_recommendations,j[2])
            tp_total += test_results[0]
            fp_total += test_results[1]
            fn_total += test_results[2]
            tn_total += test_results[3]

    elif test_type == valid_test[2]: # Determines the recommendation algorithm to be tested.
        for j in list_of_users: # For each user in the list of users, conduct a test and sum up their confusion matrix values.
            test_results = _knn_test(j[0],j[1],n_comp,n_recommendations,j[2])
            tp_total += test_results[0]
            fp_total += test_results[1]
            fn_total += test_results[2]
            tn_total += test_results[3]


    return tp_total,fp_total,fn_total,tn_total


def _check_user_list(list_of_users): # Checks if a user is unique before returning them.
    """Takes in a list of users and generates a new user id. Returns a user id who is not in the given list."""
    curr_user = pick_test_user()
    if curr_user in list_of_users:
        _check_user_list(list_of_users)

    else:
        return curr_user

    return list_of_users, curr_user



# This will take in a random test user, test book title and id and run an individual test.
# Outputs singular test values
def _b1_test(test_user, test_book_title, n_comp, n_recommendations = 10, test_book_id = None):
    """Takes in a random test user and runs an individual book_recommender_1 algorithm. Returns the calculated confusion matrix value results."""

    recommended_books = matrix_factorisation_1(test_book_title,n_recommendations,n_comp) #Stores the results of the recommendation algorithm into a list.

    df = pd.DataFrame() # Creates an empty dataframe. This is used to just to recover the book id's of the recommended books. More reliable to compare book id's (int) than book titles (string).
    for i in recommended_books:
        curr_book_row = books[books['original_title'] == i] 
        df = pd.concat([df,curr_book_row],ignore_index=True) #Stores the books in a new dataframe which contains the book titles and book id's.

    recommended_books_list = df["book_id"].values.tolist() # Creates a new list which solely stores the book id's.

    expected_books = to_read.loc[to_read['user_id'] == test_user].head(10) # Stores the first 10 books test_user wants to read.
    expected_books_list = expected_books["book_id"].values.tolist() 
   
    results = _matrix_calc(expected_books_list,recommended_books_list,test_user) #Compares the matrix calculations against 
    tp = results[0]
    fp = results[1]
    fn= results[2]
    tn = results[3]

    return tp,fp,fn,tn


# This will take in a random test user, test book title and id and run an individual test.
# Outputs singular test values
def _b2_test(test_user, test_book_title, n_comp, n_recommendations = 10, test_book_id = None):
    """Takes in a random test user and runs an individual book_recommender_1 algorithm. Returns the calculated confusion matrix value results."""

    recommended_books = matrix_factorisation_2(test_book_title,n_recommendations,n_comp) #Stores the results of the recommendation algorithm into a list.

    df = pd.DataFrame() # Creates an empty dataframe. This is used to just to recover the book id's of the recommended books. More reliable to compare book id's (int) than book titles (string).
    for i in recommended_books:
        curr_book_row = books[books['original_title'] == i] 
        df = pd.concat([df,curr_book_row],ignore_index=True) #Stores the books in a new dataframe which contains the book titles and book id's.

    recommended_books_list = df["book_id"].values.tolist() # Creates a new list which solely stores the book id's.

    expected_books = to_read.loc[to_read['user_id'] == test_user].head(10) # Stores the first 10 books test_user wants to read.
    expected_books_list = expected_books["book_id"].values.tolist() 
   
    results = _matrix_calc(expected_books_list,recommended_books_list,test_user) #Compares the matrix calculations against 
    tp = results[0]
    fp = results[1]
    fn= results[2]
    tn = results[3]

    return tp,fp,fn,tn

def _knn_test(test_user, test_book_title, n_comp, n_recommendations = 10, test_book_id = None):
    """Takes in a random test user and runs an individual book_recommender_1 algorithm. Returns the calculated confusion matrix value results."""

    recommended_books = knn_popularity_recommender(test_book_title,n_recommendations) #Stores the results of the recommendation algorithm into a list.

    df = pd.DataFrame() # Creates an empty dataframe. This is used to just to recover the book id's of the recommended books. More reliable to compare book id's (int) than book titles (string).
    for i in recommended_books:
        curr_book_row = books[books['original_title'] == i] 
        df = pd.concat([df,curr_book_row],ignore_index=True) #Stores the books in a new dataframe which contains the book titles and book id's.

    recommended_books_list = df["book_id"].values.tolist() # Creates a new list which solely stores the book id's.

    expected_books = to_read.loc[to_read['user_id'] == test_user].head(10) # Stores the first 10 books test_user wants to read.
    expected_books_list = expected_books["book_id"].values.tolist() 
   
    results = _matrix_calc(expected_books_list,recommended_books_list,test_user) #Compares the matrix calculations against 
    tp = results[0]
    fp = results[1]
    fn= results[2]
    tn = results[3]


    return tp,fp,fn,tn


def _matrix_calc (expected_books_list, recommended_books_list,test_user):
    """Takes in the expected_books list and the recommended_books list. Iterates through each, comparing both lists."""
    tp = 0
    fp = 0
    fn = 0
    tn = 0
 
    for recommended_book in recommended_books_list: # For each book in the recommended_book_list
        for expected_book in expected_books_list: # For each book in the recommended_book_list is compared against every book in the expected_books list.
            if expected_book == recommended_book: # If they match, we get a true positive.
                tp += 1
                continue

            if expected_books_list.index(expected_book) == len(expected_books_list) - 1: #If we go through every book in the expected_books_list and there is no match, that means the recommendation was wrong.
                # fp += _fp_test(recommended_book, test_user) # [OLD] We consider a false positive if we recommended a book that they expressedly showed disinterest.
                fp += 1 # Always a false positive if our recommendation does not match any of the expected books.
    
    fn = len(expected_books_list) - tp # These are items that the user wants to read but were missed by the recommender
    return tp,fp,fn,tn



# def _fp_test(recommended_book,test_user):
    """Unused function that returns a 1 or 0 based on if the user has expressed disinterest in a book."""
#     x = 0
#     rating_vals = ratings.loc[ratings['book_id'] == recommended_book]
#     if len(rating_vals) < 1: # No book was found matching the book id.
#         return x
#     else:
#         user_vals = rating_vals.loc[rating_vals['user_id'] == test_user]
#         if len(user_vals) != 1: # No user was found matching matching that book id.
#             return x
#         else:
#             y = user_vals.iloc[0,2]
#             if y < 4:
#                 x += 1
#             else:
#                 x
#     return x