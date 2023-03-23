import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle
#loading dataset using pandas 
df = pd.read_csv('lung_cancer.csv')

# droping the useless columns 
df.drop(['Name','Surname'],axis='columns',inplace=True)
inputs = df.drop('Result',axis='columns')

#seperating the target element
target = df['Result']

#spliting the dataset for training
x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

#creating the classifier
model = tree.DecisionTreeClassifier()

#fitting the model 
model.fit(x_train,y_train)

#makeing the pickel file of the model
pickle.dump(model, open("model.pkl","wb"))