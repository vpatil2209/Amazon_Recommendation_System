# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:22:33 2020

@author: vishal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")
i = input('Enter the Product Id : ')

amazon_ratings = pd.read_csv('Reviews.csv')
amazon_ratings = amazon_ratings.dropna()

popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Score'].count())
most_popular = popular_products.sort_values('Score', ascending = False)

most_popular.head(20).plot(kind = "bar")

# Subset of Amazon Ratings
amazon_ratings1 = amazon_ratings.head(10000)
ratings_utility_matrix = amazon_ratings1.pivot_table(values='Score', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()

ratings_utility_matrix.shape

X = ratings_utility_matrix.T
X.head()


from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape
X.index[432]

#i = "B00005V3DC"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID

correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape

Recommend = list(X.index[correlation_product_ID > 0.50])
Recommend.remove(i)
Recommend[0:5]


