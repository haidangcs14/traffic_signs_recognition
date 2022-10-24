import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

st.title("NHẬN DẠNG BIỂN BÁO GIAO THÔNG")

model = load_model('traffic_classifier_model.h5')

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

# Tai hinh anh len
def load_img(img_file):
    img = Image.open(img_file)
    return img

def classify(image):
    # Resize the image
    image = image.resize((30,30))
    # Inserts a new axis that will appear at the axis position in the expanded array shape
    image = np.expand_dims(image, axis=0)
    # Convert to numpy array
    image = np.array(image)
    # Make prediction
    pred = np.argmax(model.predict([image])[0])
    sign = classes[pred + 1]
    return sign

choice = st.sidebar.radio("Lựa chọn của bạn", ("Nhận dạng biển báo", "Đánh giá mô hình"))


if choice == "Nhận dạng biển báo":
    st.caption('Website này giúp bạn nhận dạng và phân loại biển báo giao thông với độ chính xác tương đối')
    st.caption('Bạn hãy chọn hình ảnh và upload vào đây :smile:!!!')

    file_uploaded = st.file_uploader("Chọn ảnh tải lên", type=["png", "jpg", "jpeg"])
    if file_uploaded is not None:
        file_details = {"filename": file_uploaded.name, "filetype": file_uploaded.type,
                        "filesize": file_uploaded.size}
        img = load_img(file_uploaded)
        st.write(file_details)
        st.image(img, caption="Ảnh bạn đã chọn")
        if st.button("Nhận dạng hình ảnh"):
            st.success("Nhận dạng thành công")
            sign = classify(img)
            st.write("Tên biển báo: ", sign)
    else:
        st.warning("Bạn cần ảnh để nhận dạng!")

elif choice == "Đánh giá mô hình":
    st.write("Kiểm tra độ chính xác của mô hình")

    st.info("Hiển thị thông tin bộ test")
    y_test = pd.read_csv('Test.csv')

    st.dataframe(y_test)
    if st.button("Đánh giá"):
        labels = y_test['ClassId'].values
        images = y_test['Path'].values
        data = []
        for img in images:
            image = Image.open(img)
            image = image.resize((30,30))
            data.append(np.array(image))

        X_test = np.array(data)
        predict = model.predict(X_test)
        predicts = np.argmax(predict, axis=1)

        st.write("Đồ thị minh họa tập nhãn của bộ test")
        st.line_chart(labels[:300])
        st.write("Đồ thị minh họa tập nhãn sau khi dự đoán")
        st.line_chart(predicts[:300])
        
        st.success("Độ chính xác của mô hình:\t\t" + str(round(accuracy_score(labels, predicts) * 100, 3)))