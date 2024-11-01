import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Danh sách thư mục chứa ảnh
folders = ["anh", "anh_1"]
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục output nếu chưa tồn tại

# Tạo file nhãn cho mỗi thư mục
for folder_index, folder in enumerate(folders):
    labels_file_path = os.path.join(output_folder, f"labels_{folder}.txt")

    # Mở file nhãn để ghi
    with open(labels_file_path, "w") as f:
        # Lặp qua các file trong thư mục
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):  # Kiểm tra file ảnh
                label = folder_index  # Gán nhãn khác nhau cho từng thư mục
                f.write(f"{filename} {label}\n")  # Ghi tên file và nhãn vào file nhãn

print("Đã tạo file nhãn trong thư mục output.")

# Danh sách file nhãn
labels_files = [os.path.join(output_folder, "labels_anh.txt"), os.path.join(output_folder, "labels_anh_1.txt")]

# Danh sách để lưu ảnh và nhãn
images = []
labels = []

# Đọc ảnh và nhãn từ từng thư mục
for folder, labels_file in zip(folders, labels_files):
    label_dict = {}
    # Đọc file nhãn
    with open(labels_file, "r") as f:
        for line in f:
            filename, label = line.strip().split()
            label_dict[filename] = int(label)

    # Tải ảnh và gán nhãn từ thư mục
    for filename in os.listdir(folder):
        if filename in label_dict:
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize tất cả ảnh về kích thước 64x64
                images.append(img.flatten())  # Chuyển ảnh thành vector
                labels.append(label_dict[filename])  # Gán nhãn từ file nhãn

# Kiểm tra nếu không có nhãn nào được gán
if len(labels) == 0:
    raise ValueError("Không có ảnh nào được gán nhãn phù hợp. Hãy kiểm tra lại các file nhãn và tên ảnh.")

# Chuyển danh sách thành mảng numpy
images = np.array(images)
labels = np.array(labels)

# Các tỷ lệ chia train-test
split_ratios = [0.8, 0.7, 0.6, 0.4]


# Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall


# Huấn luyện và đánh giá với từng tỷ lệ chia train-test
for ratio in split_ratios:
    print(f"Đánh giá với tỷ lệ train-test: {int(ratio * 100)}-{int((1 - ratio) * 100)}")

    # Chia dữ liệu theo tỷ lệ
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=1 - ratio, random_state=42)

    # SVM
    svm_model = SVC()
    svm_accuracy, svm_precision, svm_recall = train_and_evaluate_model(X_train, X_test, y_train, y_test, svm_model)
    print(f"SVM - Accuracy: {svm_accuracy:.2f}, Precision: {svm_precision:.2f}, Recall: {svm_recall:.2f}")

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_accuracy, knn_precision, knn_recall = train_and_evaluate_model(X_train, X_test, y_train, y_test, knn_model)
    print(f"KNN - Accuracy: {knn_accuracy:.2f}, Precision: {knn_precision:.2f}, Recall: {knn_recall:.2f}")
    print("-" * 40)
