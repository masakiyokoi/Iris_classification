import pandas as pd
from sklearn.model_section import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def main():
   #アヤメデータの読み込み
   iris_data = pd.read_csv("iris.csv",encoding="utf-8")

   #アヤメのデータをラベルと入力データに分離
   y = iris_data.loc[:,"Name"]
   x = iris_data.loc[:,["SepalLength","SepalLength","SepalWidth","PetalLength","PetalWidth"]]

   #データセットを学習用とテスト用に分離
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

   #学習
   clf = SVC()
   clf.fit(x_train, y_train)

   #評価する	
   y_pred = clf.predict(x_test)
   print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    main()
