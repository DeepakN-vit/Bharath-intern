import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("Netflix_data.csv",header=None)
transactions=[]
for i in range(0,7466):
  transactions.append([str(dataset.values[i,j])for j in range(0,20)])

from apyori import apriori
rules=apriori(transactions=transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)
result=list(rules)
print(result)

def inspect(results):
    movie_1         = [tuple(result[2][0][0])[0] for result in result]
    movie_2         = [tuple(result[2][0][1])[0] for result in result]
    supports    = [result[1] for result in result]

    return list(zip(movie_1, movie_2, supports))
DataFrame_intelligence = pd.DataFrame(inspect(result), columns = ['Movie1', 'Movie2', 'Support'])
DataFrame_intelligence.nlargest(n = 10, columns = 'Support')