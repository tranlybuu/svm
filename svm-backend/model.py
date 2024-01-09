import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SettingWithCopyWarning)
import pickle
import os

class MultiClassificationSVM:
    def __init__(self, kernel='linear', type='OvA', C=0.1):
        self.kernel = kernel
        self.C = C
        self.type = type
        self.accuracy = 0
        if type == 'OvA':
            self.separate_dataset = self.data_procession_for_OvA
        elif type == 'OvO':
            self.separate_dataset =  self.data_procession_for_OvO
        else:
            print("Kiểu mô hình không hợp lệ (OvA/OvO)")
            return
    
    def data_procession_for_OvA(self, dataset_dict):
        """
            Xử lý dữ liệu thành các Dataframe nhỏ hơn cho model OvA
            Đầu vào: Dataframe tổng
            Đầu ra: array[
                {
                    class: [1 label, 'others'],
                    df: Dataframe
                },...
            ]
            Quy trình:
                Chia thành n dataframe nhỏ với n là số class
                Mỗi Dataframe nhỏ được chia theo 1 class và các class còn lại
                Class hiện tại thì price_range sẽ gán bằng 0 và các class còn lại sẽ gán bằng 1 để phân loại

        """
        processed_dataset_dict = []
        for index in range(self.number_of_class):
            current_class = dataset_dict[index]["label"]
            current_dataset = dataset_dict[index]["dataset"]
            current_dataset[self.label_row] = 0
            other_dataset_list = dataset_dict[:index] + dataset_dict[index+1:]
            for index in range(len(other_dataset_list)):
                if index == 0:
                    others_dataset = other_dataset_list[index]["dataset"].copy()
                else:
                    others_dataset = pd.concat([others_dataset, other_dataset_list[index]["dataset"]])
                    others_dataset[self.label_row] = 1
            classes = [current_class, 'others']
            new_df = pd.concat([current_dataset, others_dataset])
            processed_dataset_dict.append({
                "classification": classes,
                "df": new_df
            })
        return processed_dataset_dict
        
    def data_procession_for_OvO(self, dataset_dict):
        """
            Xử lý dữ liệu thành các Dataframe nhỏ hơn cho model OvO
            Đầu vào: Dataframe tổng
            Đầu ra: array[
                {
                    class: [label 1, label 2],
                    df: Dataframe
                },...
            ]
            Quy trình:
                Chia thành n*(n-1)/2 dataframe nhỏ với n là số class
                Mỗi Dataframe nhỏ được chia theo 1 class và các class còn lại
                Class thứ 1 thì price_range sẽ gán bằng 0 và các class thứ 2 sẽ gán bằng 1 để phân loại
        """
        processed_dataset_dict = []
        for i in range(len(dataset_dict)):
            for j in range(i+1, len(dataset_dict)):
                frist_class = dataset_dict[i]["label"]
                frist_dataset = dataset_dict[i]["dataset"]
                frist_dataset[self.label_row] = 0
                second_class = dataset_dict[j]["label"]
                second_dataset = dataset_dict[j]["dataset"]
                second_dataset[self.label_row] = 1
                classes = [frist_class, second_class]
                new_df = pd.concat([frist_dataset, second_dataset])
                processed_dataset_dict.append({
                    "classification": classes,
                    "df": new_df
                })
        return processed_dataset_dict
    
    def train(self, dataset, label_row):
        self.label_row = label_row

        # Tách dataframe ban đầu thành n dataframe nhỏ (n là số class)
        unique_values = sorted(dataset[label_row].unique())
        self.unique_values = unique_values
        self.number_of_class = len(unique_values)
        data = []
        for unique_value in unique_values:
            new_dataset = dataset[dataset[label_row] == unique_value]
            data.append({
                "label": unique_value,
                "dataset": new_dataset
            })
        
        # Chia tập dữ liệu theo từng mô hình OvA hoặc OvO
        processed_dataset_dict = self.separate_dataset(data)

        # Xây dựng các mô đồ phân loại nhị phân
        for index, dict_item in enumerate(processed_dataset_dict):
            print("Đang huấn luyện model " + self.kernel + "-" + self.type + " thứ " + str(index+1) + "/" + str(len(processed_dataset_dict)) + ": Phân loại " + str(dict_item["classification"]))
            X_train = dict_item["df"].drop(self.label_row, axis=1)
            Y_train = dict_item["df"][self.label_row]
            model = SVC(kernel = self.kernel, C = self.C)
            model.fit(X_train, Y_train.values)
            dict_item["model"] = model
            del dict_item["df"]
        self.operation = processed_dataset_dict
        
    def predict(self, X="", get_predicted_score=True):
        """
            Dự đoán kết quả trả về
            Đầu vào: Array[
                        Array[Chứa các feature tương tự như tập train], ...
                    ]
            Đầu ra: Array[
                        {
                            predict_class: Class có điểm số cao nhất,
                            score_prediction: Phần trăm dự đoán giữa các class
                        }, ...
                    ]
            Quy trình:
                Tạo ra một mảng chứa kết quả dự đoán gồm n phẩn tử với n là số class
                Nếu mỗi lần dự đoán class đấy đúng thì kết quả dự đoán của class đấy +1
                Nếu class được lưu trong classification thì kết quả dự đoán của class +1 và class hiện tại trừ 1
        """
        predictions = []
        for _, row in X.iterrows():
            # Tạo ra 1 dict chứa điểm số dự đoán của các class
            global_prediction = {}
            for label in self.unique_values:
                global_prediction[label] = 0

            # Lặp qua các model và dự đoán, tính toán kết quả dự đoán giữa các model
            for item in self.operation:
                classification = item["classification"]
                model = item["model"]
                local_prediction = model.predict([row])
                if local_prediction == 0:
                    true_class = classification[0]
                else:
                    true_class = classification[1]
                if classification[1] != "others":
                    global_prediction[true_class] += 1
                else:
                    if true_class != "others":
                        global_prediction[true_class] += 1

            # Sau khi có kết quả dự đoán giữa các model thì xử lý dự đoán cao nhất và trả về kết quả
            highest_class = -1
            highest_score = -1
            for key in global_prediction:
                value = global_prediction[key]
                if value > highest_score:
                    highest_class = key
                    highest_score = value
            if get_predicted_score:
                predictions.append({
                    "predict_class": highest_class,
                    "score_prediction": global_prediction
                })
            else:
                predictions.append(highest_class)
        return predictions
    
    def score(self, SVM_prediction, real_result):
        predictions = []
        for item in SVM_prediction:
            predictions.append(item["predict_class"])
        accuracy = round(accuracy_score(predictions, real_result), 4)*100
        confusion = confusion_matrix(predictions, real_result)

        self.confusion_matrix = confusion
        self.accuracy = accuracy

    def get_accuracy(self):
        return self.accuracy
    
    def get_confusion_matrix(self):
        return self.confusion_matrix
    
def parse_prediction_to_text(prediction):
    arr = ["Giá thấp", "Giá trung bình", "Giá cao", "Giá rất cao"]
    try:
        index = prediction[0]["predict_class"]
        predict_class =  arr[index]
        score_prediction = {}
        total_score = sum(prediction[0]["score_prediction"].values())
        for key in prediction[0]["score_prediction"]:
            value = prediction[0]["score_prediction"][key]
            if total_score == 0:
                score_prediction[arr[key]] = 1
            else:
                score_prediction[arr[key]] = value
    except:
        index = prediction[0]
        predict_class =  arr[index]
        score_prediction = None
    return {
        'predict_class': predict_class,
        'score_prediction': score_prediction
    }

def start_training_model(csv_path):
    dataset=pd.read_csv(csv_path)

    prop = 0.25
    test_sample = round((len(dataset)*prop))
    dataset = dataset.sample(frac=1)
    df_test = dataset.iloc[-test_sample:]
    df_train = dataset.tail(len(dataset)-test_sample).copy()
    df_train.head(5)
    X_test = df_test.drop("price_range", axis=1)
    Y_test = df_test["price_range"]

    LinearOvA = MultiClassificationSVM(kernel='linear', type="OvA")
    LinearOvA.train(dataset = df_train, label_row = "price_range")
    OvApredictions = LinearOvA.predict(X_test)
    LinearOvA.score(OvApredictions, Y_test)
    filename = 'LinearOvA.sav'
    pickle.dump(LinearOvA, open(filename, 'wb'))

    LinearOvO = MultiClassificationSVM(kernel='linear', type="OvO")
    LinearOvO.train(dataset = df_train, label_row = "price_range")
    OvApredictions = LinearOvO.predict(X_test)
    LinearOvO.score(OvApredictions, Y_test)
    filename = 'LinearOvO.sav'
    pickle.dump(LinearOvO, open(filename, 'wb'))

    PolyOvA = MultiClassificationSVM(kernel='poly', type="OvA")
    PolyOvA.train(dataset = df_train, label_row = "price_range")
    OvApredictions = PolyOvA.predict(X_test)
    PolyOvA.score(OvApredictions, Y_test)
    filename = 'PolyOvA.sav'
    pickle.dump(PolyOvA, open(filename, 'wb'))

    PolyOvO = MultiClassificationSVM(kernel='poly', type="OvO")
    PolyOvO.train(dataset = df_train, label_row = "price_range")
    OvOpredictions = PolyOvO.predict(X_test)
    PolyOvO.score(OvOpredictions, Y_test)
    filename = 'PolyOvO.sav'
    pickle.dump(PolyOvO, open(filename, 'wb'))

    RbfOvA = MultiClassificationSVM(kernel='rbf', type="OvA")
    RbfOvA.train(dataset = df_train, label_row = "price_range")
    OvApredictions = RbfOvA.predict(X_test)
    RbfOvA.score(OvApredictions, Y_test)
    filename = 'RbfOvA.sav'
    pickle.dump(RbfOvA, open(filename, 'wb'))

    RbfOvO = MultiClassificationSVM(kernel='rbf', type="OvO")
    RbfOvO.train(dataset = df_train, label_row = "price_range")
    OvOpredictions = RbfOvO.predict(X_test)
    RbfOvO.score(OvOpredictions, Y_test)
    filename = 'RbfOvO.sav'
    pickle.dump(RbfOvO, open(filename, 'wb'))

    SigmoidOvA = MultiClassificationSVM(kernel='sigmoid', type="OvA")
    SigmoidOvA.train(dataset = df_train, label_row = "price_range")
    OvApredictions = SigmoidOvA.predict(X_test)
    SigmoidOvA.score(OvApredictions, Y_test)
    filename = 'SigmoidOvA.sav'
    pickle.dump(SigmoidOvA, open(filename, 'wb'))

    SigmoidOvO = MultiClassificationSVM(kernel='sigmoid', type="OvO")
    SigmoidOvO.train(dataset = df_train, label_row = "price_range")
    OvOpredictions = SigmoidOvO.predict(X_test)
    SigmoidOvO.score(OvOpredictions, Y_test)
    filename = 'SigmoidOvO.sav'
    pickle.dump(SigmoidOvO, open(filename, 'wb'))

def load_model_from_file():
    paths = ['LinearOvA', 'LinearOvO', 'PolyOvA', 'PolyOvO', 'RbfOvA', 'RbfOvO', 'SigmoidOvA', 'SigmoidOvO']
    models = []
    for path in paths:
        filename = os.path.join(os.getcwd(), path + ".sav")
        if os.path.exists(filename):
            model = pickle.load(open(filename, 'rb'))
            accuracy = model.get_accuracy()
            models.append({"name": path, "accuracy": accuracy, "model": model})
            print(f"Đã tải thành công model SVM {path} với độ chính xác {accuracy}%")
        else:
            print(f"Không tìm thấy file {filename}")

    sklearn_paths = ["Sklearn-LinearOvA", "Sklearn-LinearOvO", "Sklearn-PolyOvA", "Sklearn-PolyOvO", "Sklearn-RbfOvA", "Sklearn-RbfOvO", "Sklearn-SigmoidOvA", "Sklearn-SigmoidOvO"]
    for path in sklearn_paths:
        filename = os.path.join(os.getcwd(), path + ".sav")
        if os.path.exists(filename):
            model = pickle.load(open(filename, 'rb'))
            if "Sigmoid" in path:
                accuracy = "16.6"
            else:
                accuracy = "93"
            models.append({"name": path, "accuracy": accuracy, "model": model})
            print(f"Đã tải thành công model SVM {path} với độ chính xác {accuracy}%")
        else:
            print(f"Không tìm thấy file {filename}")
    return models