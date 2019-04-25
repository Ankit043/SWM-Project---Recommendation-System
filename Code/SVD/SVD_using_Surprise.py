from surprise import Reader, Dataset, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import warnings

def read_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ratings.dat', sep='::', names=names)
    df.head()
    df = df.loc[:,['user_id', 'item_id', 'rating']]
    return df

def create_user_item_matrix(df):
    users = df.user_id.unique().shape[0]
    items = df.item_id.unique().shape[0]
    u_i_matrix = np.zeros((users, items))
    for row in df.itertuples():
        if row[1] < 6040 and row[2] < 3706:
            u_i_matrix[row[1] - 1, row[2] - 1] = row[3]
    return u_i_matrix

warnings.filterwarnings("ignore")

rd = Reader()

ratings = [i.strip().split("::") for i in open('ratings.dat', 'r').readlines()]
ratings_df = pd.DataFrame(ratings, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
data = Dataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], rd)

algo = SVD()

test_size = [0.1,0.15,0.2,.25]
best_rmse = float("inf")
rmse_list = []
for i in test_size:
    trainset, testset = train_test_split(data, test_size=i)

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)
    # RMSE
    rmse = accuracy.rmse(predictions, verbose = False)
    print("RMSE for test train split of {0}, is {1}".format(i,rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        optimal_train_test_split = i
    rmse_list.append(rmse)

print("Best rmse value is {1}, for a train_test_split of {0}".format(best_rmse,optimal_train_test_split))

test_size = np.array(test_size)
rmse_list = np.array(rmse_list)
plt.xlabel('Train_Test Split Ratio')
plt.ylabel('MSE')
plt.plot(test_size,rmse_list,'rv--')
plt.show()

trainset, testset = train_test_split(data, test_size=optimal_train_test_split)
algo.fit(trainset)
predictions = algo.test(testset)
rmse_optimal = accuracy.rmse(predictions, verbose= False)

#Predicting movie ratings for a user
uid = str(196)
iid = str(302)

df = read_data()
user_item_matrix = create_user_item_matrix(df)

actual_user_rating = user_item_matrix[198][302]

#Prediction for specific user and item
pred = algo.predict(uid, iid, r_ui=actual_user_rating, verbose=False)
print("For user {0} and movie {1}, the actual rating is {2} and the predicted rating is {3}".format(uid,iid,actual_user_rating,pred[3]))





