from sklearn.metrics import accuracy_score
from pandas import read_csv

nn = read_csv('output_nn.csv',index_col=0)
svm = read_csv('output_svm.csv',index_col=0)
y81 = read_csv('ylabel_nn_0_8218.csv',index_col=0)


print("nn:",accuracy_score(y81,nn))
print("svm:",accuracy_score(y81,svm))