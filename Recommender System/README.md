
# Recommendation systems
Build a recommendation systems from strach using collborative filtering with matrix factorization


## The Cold Start Problem

The colaborative filtering method discussed in class does not address the problem of new user or new movies. What prediction would you use in these cases:

* A new user but a known movie
* A new movie and a known user
* A new user and new movie

Possible Solutions:

* Use median of ratings for the known movie. Alternatively ask the new user to rate some movies and use cluster method or simply a cosine similarity to assign the user to a group and use the group average for the known movie instead.

* Use genre, cast, directors etc to assign the new movie to a group of similar kind that the user has rated, then used the group average of the movie genre to rate the new movie.

* A combination of two methods above, ask the new user to rate some movies then assign she/he to a group, then cluster the new movie and use average rating of user-group for the movie-cluster as the prediction.


# Collaborative Filtering with Stochastic Gradient Descent

Build a collaborative filtering model to predict Netflix ratings.

- Build the general architecture of a learning algorithm, including:
    - Encoding rating data
    - Initializing parameters
    - Calculating the cost function
    - Calculating gradient
    - Using an optimization algorithm (gradient descent)
    - Predicting on new data
- Putting it all together.


```python
import numpy as np
import pandas as pd
```

## Encoding rating data
Here are our very small subset of fake data to get us started.


```python
# The first row says that user 1 reated movie 11 with a score of 4
!cat tiny_training2.csv
```

    userId,movieId,rating
    11,1,4
    11,23,5
    2,23,5
    2,4,3
    31,1,4
    31,23,4
    4,1,5
    4,3,2
    52,1,1
    52,3,4
    61,3,5
    7,23,1
    7,3,3



```python
# here is a handy function from fast.ai
def proc_col(col):
    """Encodes a pandas column with continous ids.
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)
```


```python
def encode_data(df):
    """Encodes rating data with continous user and movie ids using
    the helpful fast.ai function from above.

    Arguments:
      train_csv: a csv file with columns user_id,movie_id,rating

    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies

    """
    # YOUR CODE HERE
    df['userId'] = proc_col(df['userId'])[1]
    num_users = proc_col(df['userId'])[2]
    df['movieId'] = proc_col(df['movieId'])[1]
    num_movies = proc_col(df['movieId'])[2]

    return df, num_users, num_movies
```


```python
df = pd.read_csv("tiny_training2.csv")
df, num_users, num_movies = encode_data(df)
```


```python
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
assert(num_users == 7)
```


```python
assert(num_movies == 4)
```


```python
np.testing.assert_equal(df["userId"].values, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6]))
```

## Initializing parameters


```python
def create_embedings(n, K):
    """ Create a numpy random matrix of shape n, K

    The random matrix should be initialized with uniform values in (0, 6/K)
    Arguments:

    Inputs:
    n: number of items/users
    K: number of factors in the embeding

    Returns:
    emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6*np.random.random((n, K)) / K
    return emb

# here is an example on how the prediction matrix would look like with 7 users and 5 movies
np.dot(create_embedings(7,3), create_embedings(5,3).transpose())
```




    array([[ 3.55790894,  4.69774849,  0.92361109,  1.58739544,  3.00593239],
           [ 4.69774849,  7.44656163,  1.18135616,  2.64524868,  4.74559066],
           [ 0.92361109,  1.18135616,  0.24548062,  0.34025121,  0.69616965],
           [ 1.58739544,  2.64524868,  0.34025121,  1.61561   ,  2.41361975],
           [ 3.00593239,  4.74559066,  0.69616965,  2.41361975,  3.82505541],
           [ 2.02000808,  3.29656257,  0.43174569,  2.065911  ,  3.07264619],
           [ 2.07691001,  3.02887291,  0.53270924,  1.02482544,  1.90251125]])



## Encoding Y as a sparse matrix
This code helps you encode a $Y$ as a sparse matrix from the dataframe.


```python
from scipy import sparse
def df2matrix(df, nrows, ncols, column_name="rating"):
    """ Returns a sparse matrix constructed from a dataframe

    This code assumes the df has columns: MovieID,UserID,Rating
    """
    values = df[column_name].values
    ind_movie = df['movieId'].values
    ind_user = df['userId'].values
    return sparse.csc_matrix((values,(ind_user, ind_movie)),shape=(nrows, ncols))
```


```python
df = pd.read_csv("tiny_training2.csv")
df, num_users, num_movies = encode_data(df)
Y = df2matrix(df, num_users, num_movies)
```


```python
print(Y)
```

      (0, 0)	4
      (2, 0)	4
      (3, 0)	5
      (4, 0)	1
      (0, 1)	5
      (1, 1)	5
      (2, 1)	4
      (6, 1)	1
      (1, 2)	3
      (3, 3)	2
      (4, 3)	4
      (5, 3)	5
      (6, 3)	3



```python
def sparse_multiply(df, emb_user, emb_movie):
    """ This function returns U*V^T element wise multi by R as a sparse matrix.

    It avoids creating the dense matrix U*V^T
    """
    df["Prediction"] = np.sum(emb_user[df["userId"].values]*emb_movie[df["movieId"].values], axis=1)
    return df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction")
```

## Calculating the cost function


```python
# Use vectorized computation for this function. No loops!
# Hint: use df2matrix and sparse_multiply
def cost(df, emb_user, emb_movie):
    """ Computes mean square error

    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]

    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies

    Returns:
      error(float): this is the MSE
    """
    # YOUR CODE HERE
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    pred = sparse_multiply(df, emb_user, emb_movie)
    e = Y-pred
    error = e.multiply(e).sum()/Y.nnz
    return error
```


```python
emb_user = np.ones((num_users, 3))
emb_movie = np.ones((num_movies, 3))
error = cost(df, emb_user, emb_movie)
assert(np.around(error, decimals=2) == 2.23)
```

## Calculating gradient


```python
def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """ Computes finite difference on MSE(U, V).

    This function is used for testing the gradient function.
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1)/e
```


```python
def gradient(df, Y, emb_user, emb_movie):
    """ Computes the gradient.

    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]

    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies

    Returns:
      d_emb_user
      d_emb_movie
    """
    # YOUR CODE HERE
    pred = sparse_multiply(df, emb_user, emb_movie)
    Delta = Y-pred
    d_emb_user = -2/Y.nnz * Delta * emb_movie
    d_emb_movie= -2/Y.nnz * Delta.transpose() * emb_user

    return d_emb_user, d_emb_movie
```


```python
K = 3
emb_user = create_embedings(num_users, K)
emb_movie = create_embedings(num_movies, K)
Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
```


```python
user=1
approx = np.array([finite_difference(df, emb_user, emb_movie, ind_u=user, k=i) for i in range(K)])
assert(np.all(np.abs(grad_user[user] - approx) < 0.0001))
```


```python
movie=1
approx = np.array([finite_difference(df, emb_user, emb_movie, ind_m=movie, k=i) for i in range(K)])
assert(np.all(np.abs(grad_movie[movie] - approx) < 0.0001))
```

## Using gradient descent with momentum


```python
# you can use a for loop to iterate through gradient descent
def gradient_descent(df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number of iterations.

    Prints training cost and validation cost (if df_val is not None) every 50 iterations.

    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    # YOUR CODE HERE
    beta = .9 # momentum
    grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
    V_grad_user, V_grad_movie = grad_user, grad_movie # V_0 initialized
    for i in range(iterations):   
        grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
        V_grad_user = beta * V_grad_user + (1-beta) * grad_user
        V_grad_movie = beta * V_grad_movie + (1-beta) * grad_movie
        emb_user -= learning_rate * V_grad_user
        emb_movie -= learning_rate * V_grad_movie
        if (i+1)%50==0:
            print("Training cost: ", cost(df, emb_user, emb_movie))
            if df_val is not None:
                print("Validation cost: ", cost(df_val, emb_user, emb_movie))
    return emb_user, emb_movie
```


```python
emb_user = create_embedings(num_users, 3)
emb_movie = create_embedings(num_movies, 3)
emb_user, emb_movie = gradient_descent(df, emb_user, emb_movie, iterations=200, learning_rate=0.01)
```

    Training cost:  1.70136430619
    Training cost:  0.974870517206
    Training cost:  0.699145565693
    Training cost:  0.526562109415



```python
train_mse = cost(df, emb_user, emb_movie)
assert(np.around(train_mse, decimals=2) == 0.53)
```

## Predicting on new data
Now we should write a function that given new data is able to predict ratings. First we write a function that encodes new data. If a new user or item is present that row should be remove. Collaborative Filtering is not good at handling new users or new items. To help with this task, you could write a an auxiliary function similar to `proc_col`.


```python
def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    # YOUR CODE HERE
    # drop new users and/or new movies
    movieId_list = set(df_val.movieId.values) & set(df_train.movieId.values)
    userId_list = set(df_val.userId.values) & set(df_train.userId.values)

    df_val = df_val[df_val.movieId.isin(movieId_list)]
    df_val = df_val[df_val.userId.isin(userId_list)]

    # same encoding
    IDs = proc_col(df_train.userId)[0]
    Movies = proc_col(df_train.movieId)[0]

    df_val.userId = np.array([IDs[x] for x in df_val.userId])
    df_val.movieId = np.array([Movies[x] for x in df_val.movieId])

    return df_val
```


```python
df_t = pd.read_csv("tiny_training2.csv")
df_v = pd.read_csv("tiny_val2.csv")
df_v = encode_new_data(df_v, df_t)
```


```python
assert(len(df_v.userId.unique())==2)
```


```python
assert(len(df_v) == 2)
```

## Putting it all together
For this part you should get data from here
`wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip`


```python
# Don't change this path use a simlink if you have the data somewhere else
path = "ml-latest-small/"
data = pd.read_csv(path + "ratings.csv")
# sorting by timestamp take as validation data the most recent data doesn't work so let's just take 20%
# at random
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
val = data[~msk].copy()
df_train, num_users, num_movies = encode_data(train.copy())
df_val = encode_new_data(val.copy(), train.copy())
print(len(val), len(df_val))
```

    20205 19507



```python
K = 50
emb_user = create_embedings(num_users, K)
emb_movie = create_embedings(num_movies, K)
emb_user, emb_movie = gradient_descent(df_train, emb_user, emb_movie, iterations=2000, learning_rate=1, df_val=df_val)
```

    Training cost:  9.98836344217
    Validation cost:  10.1253120833
    Training cost:  7.22565025945
    Validation cost:  7.36167540124
    ......
    Training cost:  0.775129612323
    Validation cost:  0.916282997144
    Training cost:  0.766573002646
    Validation cost:  0.910427187685



```python
train_mse = cost(df_train, emb_user, emb_movie)
val_mse = cost(df_val, emb_user, emb_movie)
print(train_mse, val_mse)
```

    0.766573002646 0.910427187685



```python
train_mse = cost(df_train, emb_user, emb_movie)
assert(np.around(train_mse, decimals=2) == 0.77)
```


```python
val_mse = cost(df_val, emb_user, emb_movie)
assert(np.around(val_mse, decimals=2) == 0.91)
```
