#每次traint出來結果都不盡相同
from numpy import*
from sklearn import datasets
from sklearn.cross_validation import train_test_split
#cross validation : 交叉驗證
from sklearn.neighbors import KNeighborsClassifier  #最近鄰居法(K-近鄰以算法)
#k-neighbors : 測試項會被歸類在最靠近它的k個點中，最頻繁出現的類別項。
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC #support vector classify

iris = datasets.load_iris() #sklearn的database有鳶尾花的data
iris_x = iris.data
iris_y = iris.target #欲訓練的答案，target 目標
#print("x : ", iris_x[:, :2]) x[:, :]
#將x視為一個二維陣列
#第一個冒號是要列出x裡面的幾個元素，第二個是要印出x每個元素裡面的幾個子元素(x的每個元素裡面都有四個子元素)

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)
#test data 比例佔30%

#看iris的資訊
print(iris.keys())
print("DESCR: ", iris.DESCR) #顯示iris data的詳細資訊
print("target names: ", iris.target_names)
print("feature names: ",iris.feature_names)


#用knn演算法
knn = KNeighborsClassifier()
knn.fit(x_train, y_train) #放入要training的data
f=knn.predict(x_test)
#print(knn.predict(x_test)) #預測的數
count=0
for i in range(0, len(y_test)):
    if (f[i]==y_test[i]):
        count+=1
print("kn score : ", count/len(y_test))

#用decision tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print("dt score : ", dt.score(x_test, y_test))

#用SVC演算法
clf = SVC()
clf.fit(x_train, y_train)
print("SVC score : ", clf.score(x_test, y_test))