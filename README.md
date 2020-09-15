#  Task_ds-ml
# To Explore Decision Tree Algorithm
For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly. 

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

#Loading the iris dataset
iris=load_iris()

df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
x=df


y=iris.target
print(y)

#Splitting the iris Dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#DECISION TREE MODEL
tree_clf= DecisionTreeClassifier()
model=tree_clf.fit(x,y)

#Comparing Actual Data with Predicted Data
y_pred=model.predict(x_test)
df1=pd.DataFrame({'Actual Data':y_test,'Predicted Data':y_test})
print(df1.head())

#Displaying Decision Tree
dot_data=export_graphviz(tree_clf,out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,special_characters=True)
graph=graphviz.Source(dot_data)
graph.render("iris")

#Testing the model if it would be able to predict the right class accordingly for sample data
sampledata=[3,4,3,2]
result=model.predict([sampledata])
print("The predicted class for sample data is" ,result)
