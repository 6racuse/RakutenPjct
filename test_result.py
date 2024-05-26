from sklearn.metrics import accuracy_score
from pandas import read_csv

nn = read_csv('./output/output_nn.csv',index_col=0)
svm = read_csv('./output/output_svm.csv',index_col=0)
y81 = read_csv('./output/ylabel_nn_0_8218.csv',index_col=0)
knn = read_csv('./output/output_knn.csv', index_col=0)
weighted_vote = read_csv('./output/output_weighted_vote.csv', index_col=0)


print("nn:",accuracy_score(y81,nn))
print("svm:",accuracy_score(y81,svm))
print("knn:", accuracy_score(y81,knn))
print("weighted_vote", accuracy_score(y81, weighted_vote))