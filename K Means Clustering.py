'''
Program to implement K-Means Clustering Algorithm.
Developed by: SRIJITH R
RegisterNumber: 212221240054
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

x1 = pd.read_csv('clustering.csv')
print(x1.head(2))
x2 = x1.loc[:, ['ApplicantIncome','LoanAmount']]
print(x2.head(2))

x = x2.values
sns.scatterplot(x[:,0], x[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmean =KMeans(n_clusters=3)
kmean.fit(x)

print('Clusters Centers:', kmean.cluster_centers_)
print('Labels:', kmean.labels_)

predicted_class = kmean.predict([[1000,100]])
print('The cluster group for Applicant Income 1000 and Loanamount 100 :' ,predicted_class)