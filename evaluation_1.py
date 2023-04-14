from recommender_system import matrix_factorisation_1
import pandas as pd
import numpy as np
import random

book_title = 'Memoirs of a Geisha'
n_recommendations = 20



# print(matrix_factorisation_1(book_title,n_recommendations))


# books = pd.read_csv('Goodbooks-10k Dataset/books.csv', sep=',')
# books = books.iloc[:, :16] # Splices the first 16 columns.
# books = books.drop(columns=['title', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'original_publication_year','language_code','work_ratings_count','work_text_reviews_count'])
#
#
# ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')


to_read = pd.read_csv('Goodbooks-10k Dataset/to_read.csv', sep = ',')
max_users = to_read.max()[0] # 53424 users, however, in the to_read dataset only 48871 users rated books.





# check_user() randomly picks a user who wants to read at least 10 books.
def check_user(x = 0): # Set default to 0 to avoid argument confusion.
    random_user = random.randint(0, max_users) #Generates a random number, capped at the maximum users.
    duplicate_count = to_read.pivot_table(index = ['user_id'], aggfunc ='size') #Creates a pivot table around users who have read multiple books.
    a = duplicate_count.mean() # On average, most users have rated at least 18 books.
    chosen_user = x #Recursively stores the chosen_user

    if random_user in duplicate_count.index:
        if duplicate_count[random_user] <= 9:
            return check_user(random_user) # If user has not read more than 10 books, repeat.
        if duplicate_count[random_user] >= 10:
            chosen_user = random_user # Success: If user has read more than 10 books
    else:
        return check_user(random_user) # If the user is not in the index, repeat.

    return chosen_user



def pick_test_user():
    ratings = pd.read_csv('Goodbooks-10k Dataset/ratings.csv', sep=',')
    test_user = check_user() #Picks a random user
    a = ratings.loc[ratings['user_id'] == test_user] #Isolates all the books the test_user has rated.
    a = a.sort_values(by = ['rating'], ascending = False)
    list_a = list(a.iloc[0]) # Stores the highest rated book into a list.
    test_book_id = list_a[1]
    test_book_rating = list_a[2]

    return test_user, test_book_id, test_book_rating

print(pick_test_user())
