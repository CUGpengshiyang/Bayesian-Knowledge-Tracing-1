import numpy as np
from sklearn.metrics import roc_auc_score
import csv

acc = 0
#currently only accounts for 4-fold cross validation data
for index in range(4):
    with open('results' + str(index) + '.csv', 'rU') as f:
	   reader = csv.reader(f)
	   user_y = []
	   user_s = []
	   for i, row in enumerate(reader):
		  if i == 0:
			continue
		  if len(row) is not 0:
			user_y.append(int(row[0]))
			user_s.append(float(row[1]))
	   y_true = np.array(user_y)
	   y_scores = np.array(user_s)
	   user_auc = roc_auc_score(y_true, y_scores)
    print "auc of " + str(index) + ": "+ str(user_auc)
    acc += user_auc

print "average auc: " + str(acc/4)
