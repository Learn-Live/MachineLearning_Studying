from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from sklearn_issues.numpy_load import load_npy_data


def main_xgboost(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    listdir = {}
    for i in range(0 ,len(y_test)):
        if y_test[i] not in listdir:
            listdir.update({y_test[i] :0})
        else:
            listdir[y_test[i]] = listdir[y_test[i]] +1
    print(listdir)
    listdir = {}
    for i in range(0, len(y_train)):
        if y_train[i] not in listdir:
            listdir.update({y_train[i]: 0})
        else:
            listdir[y_train[i]] = listdir[y_train[i]] + 1
    print(listdir)

    # train&test
    result = []
    # for i in range(10,300,30):
    #    value.append(i)
    # value = [100]
    value = 500
    print("n_estimators: ", value)
    truncatelist = [2500]
    # for i in range(50,1500,50):
    for i in truncatelist:
        print(i)
        # clf = RandomForestClassifier(n_estimators=value, min_samples_leaf=2)
        clf = XGBClassifier(n_estimators=150)
        X_train_t = X_train[:, :i]

        X_test_t = X_test[:, :i]
        print("before input....")
        print(X_train_t.shape)
        print(y_train.shape)
        print(X_test_t.shape)
        print(y_test.shape)
        print((X_train_t[0])[0:10])
        clf.fit(X_train_t, y_train)
        predtest = clf.predict(X_test_t)
        result.append(metrics.accuracy_score(y_test, predtest))
        print(confusion_matrix(y_test, predtest))
        predtrain = clf.predict(X_train_t)
        print(confusion_matrix(y_train, predtrain))
        print("train acc:", metrics.accuracy_score(y_train, predtrain))
        print("test acc", metrics.accuracy_score(y_test, predtest))

        print(result)
    print(result)

if __name__ == '__main__':
    input_file = '../input_data/trdata-8000B.npy'
    X, y = load_npy_data(input_file)
    main_xgboost(X, y)