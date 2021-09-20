import numpy as np 
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

dataset_x = np.load('face_enoding.npy')
dataset_y = np.load('face_names_labels.npy')

dataset_x = normalize(dataset_x,norm='l2')



#(cross validation to model selection)
X_train, X_test, Y_train, Y_test = train_test_split(dataset_x, dataset_y, test_size = 0.2, random_state = 2)

# fit model
model = SVC(kernel='linear', probability=True)
#model.fit(X_train, Y_train)
model.fit(dataset_x, dataset_y)



Y_pred=model.predict(X_test)
cm= confusion_matrix(Y_test, Y_pred)

  