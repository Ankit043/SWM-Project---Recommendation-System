import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import warnings

def read_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ratings.dat', sep='::', names=names)
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

def calculate_sparsity(u_i_matrix):
    sparsity = float(len(u_i_matrix.nonzero()[0]))
    sparsity /= (u_i_matrix.shape[0] * u_i_matrix.shape[1])
    sparsity *= 100
    sparsity = 100 - sparsity
    print('Sparsity: {:4.2f}%'.format(sparsity))


def train_test_split(ratings, split_ratio):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        length = len(ratings[user, :].nonzero()[0])
        length = (length * split_ratio)//100
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=length,
                                        replace=False)
        train[user, test_ratings] = 0
        test[user, test_ratings] = ratings[user, test_ratings]

    return train, test

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    if kind == 'item':
        similarity = ratings.T.dot(ratings) + epsilon
    elif kind == 'user':
        similarity = ratings.dot(ratings.T) + epsilon
    norm = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / norm / norm.T)

def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    elif kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return math.sqrt(mean_squared_error(pred, actual))

warnings.filterwarnings("ignore")
df = read_data()
u_i_matrix = create_user_item_matrix(df)
calculate_sparsity(u_i_matrix)

test_size = [0.1,0.15,0.2,.25]
best_rmse_user = float("inf")
best_rmse_item = float("inf")
rmse_user_list = []
rmse_item_list= []
for i in test_size:
    train, test = train_test_split(u_i_matrix,int(i*100))

    user_similarity = fast_similarity(train, kind='user')
    item_similarity = fast_similarity(train, kind='item')

    item_prediction = predict_fast_simple(train, item_similarity, kind='item')
    user_prediction = predict_fast_simple(train, user_similarity, kind='user')

    rmse_user = get_mse(user_prediction, test)
    print("User Based - RMSE for test train split of {0}, is {1}".format(i, rmse_user))
    if rmse_user < best_rmse_user:
        best_rmse_user = rmse_user
        optimal_train_test_split_user = i
        best_user_prediction = user_prediction
    rmse_user_list.append(rmse_user)

    rmse_item = get_mse(item_prediction, test)
    print("Item Based - RMSE for test train split of {0}, is {1}".format(i, rmse_item))
    if rmse_item < best_rmse_item:
        best_rmse_item = rmse_item
        optimal_train_test_split_item = i
        best_item_prediction = item_prediction
    rmse_item_list.append(rmse_item)

print("User Based - Best rmse value is {0}, "
      "for a train_test_split of {1}".format(best_rmse_user,optimal_train_test_split_user))
print("Item Based - Best rmse value is {0}, "
      "for a train_test_split of {1}".format(best_rmse_item,optimal_train_test_split_item))

test_size = np.array(test_size)
rmse_user_list = np.array(rmse_user_list)
plt.xlabel('Train_Test Split Ratio')
plt.ylabel('User Based - MSE')
plt.plot(test_size,rmse_user_list,'rv--')
plt.show()

rmse_item_list = np.array(rmse_item_list)
plt.xlabel('Train_Test Split Ratio')
plt.ylabel('User Based - MSE')
plt.plot(test_size,rmse_item_list,'rv--')
plt.show()

uid = 198
iid = 302

print("User Based - For user {0} and movie {1}, the actual rating is {2} and the "
      "predicted rating is {3}".format(uid,iid,u_i_matrix[uid][iid],best_user_prediction[uid][iid]))

print("Item Based - For user {0} and movie {1}, the actual rating is {2} and the "
      "predicted rating is {3}".format(uid,iid,u_i_matrix[uid][iid],best_item_prediction[uid][iid]))




