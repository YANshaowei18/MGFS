import pandas as pd
import numpy as np
import copy
import warnings
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import KFold,cross_val_score
warnings.filterwarnings('ignore')

def data_encode_normalize(origin_data):
    # origin_data must be pandas
    label_name = origin_data.columns[-1]
    dataset = origin_data.iloc[:,0:-1]
    attr_names = list(dataset.columns)
    # feature dimensions greater than 100 default to numeric data
    if(len(attr_names)>100):
        dataset[attr_names] = MinMaxScaler().fit_transform(dataset[attr_names])
    else:
        nominal_list = []
        numerical_list = []
        for i,name in enumerate(attr_names):
            if(np.issubdtype(dataset[name],np.number)==True):
                numerical_list.append(name)
            else:
                nominal_list.append(name)
        if(len(numerical_list)!=0):
            dataset[numerical_list] = MinMaxScaler().fit_transform(dataset[numerical_list])
        for j,name in enumerate(nominal_list):
            encoder = OneHotEncoder()
            x_encode = encoder.fit_transform(dataset[[name]]).toarray()
            dataset = pd.concat([dataset,pd.DataFrame(x_encode,columns=encoder.get_feature_names_out([name]))],axis=1)
            dataset.drop(name,axis=1,inplace=True)
            del encoder,x_encode
    dataset[label_name] = LabelEncoder().fit_transform(origin_data[label_name])
    return dataset


def Classification_evaluation(data,S):

        item_score_knn = []
        item_score_dt = []
        item_score_gnb = []

        res_score_knn = []
        res_score_dt = []
        res_score_gnb = []

        for i in range(len(S)):
            attr_red = S[0:i + 1]
            origin_data_red = data[np.array(list(data))[attr_red]]
            origin_data_red['Class'] = data.iloc[:, -1]
            data_red = data_encode_normalize(origin_data=origin_data_red)
            x_red = data_red.values[:, :-1]
            y = data_red.values[:, -1]
            for j in range(10):
                model_knn = KNeighborsClassifier(n_neighbors=3)
                model_dt = DecisionTreeClassifier()
                model_gnb = GaussianNB()
                cv = KFold(n_splits=10, shuffle=True)
                score_knn = cross_val_score(model_knn, x_red, y, cv=cv)
                score_dt = cross_val_score(model_dt, x_red, y, cv=cv)
                score_gnb = cross_val_score(model_gnb, x_red, y, cv=cv)
                item_score_knn.append(np.mean(score_knn))
                item_score_dt.append(np.mean(score_dt))
                item_score_gnb.append(np.mean(score_gnb))
                del model_knn,model_dt,model_gnb,score_knn,score_dt,score_gnb,cv

            res_score_knn.append(np.mean(np.array(item_score_knn)))
            res_score_dt.append(np.mean(np.array(item_score_dt)))
            res_score_gnb.append(np.mean(np.array(item_score_gnb)))
            item_score_knn.clear()
            item_score_dt.clear()
            item_score_gnb.clear()

        knn_opti_fea = S[:np.argmax(res_score_knn)]
        dt_opti_fea = S[:np.argmax(res_score_dt)]
        gnb_opti_fea = S[:np.argmax(res_score_gnb)]

        ori_score_knn = []
        ori_score_dt = []
        ori_score_gnb = []
        data_ = data_encode_normalize(origin_data=data)
        X = data_.values[:, :-1]
        y = data_.values[:, -1]
        for z in range(10):
            model_knn = KNeighborsClassifier(n_neighbors=3)
            model_dt = DecisionTreeClassifier()
            model_gnb = GaussianNB()
            cv = KFold(n_splits=10, shuffle=True)
            score_knn = cross_val_score(model_knn, X, y, cv=cv)
            score_dt = cross_val_score(model_dt, X, y, cv=cv)
            score_gnb = cross_val_score(model_gnb, X, y, cv=cv)
            ori_score_knn.append(np.mean(score_knn))
            ori_score_dt.append(np.mean(score_dt))
            ori_score_gnb.append(np.mean(score_gnb))
            del model_knn, model_dt, model_gnb, score_knn, score_dt, score_gnb, cv


        print(f"Original KNN ACC: {np.mean(np.array(ori_score_knn))*100:.4f}")
        print(f"Original DT ACC: {np.mean(np.array(ori_score_dt))*100:.4f}")
        print(f"Original GNB ACC: {np.mean(np.array(ori_score_gnb))*100:.4f}")

        print(f"Optimal KNN ACC: {max(res_score_knn)*100:.4f}")
        print(f"Optimal DT ACC: {max(res_score_dt)*100:.4f}")
        print(f"Optimal GNB ACC: {max(res_score_gnb)*100:.4f}")

        print("Optimal KNN Feature Subset: {}".format(knn_opti_fea))
        print("Optimal DT Feature Subset: {}".format(dt_opti_fea))
        print("Optimal GNB Feature Subset: {}".format(gnb_opti_fea))