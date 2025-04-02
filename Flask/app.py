from flask import Flask, render_template, flash, request, url_for, redirect, session
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def home(): 
    return render_template("index.html")
@app.route('/image',methods=['GET','POST'])
@app.route('/image#result',methods=['GET','POST'])
def image():
    accuracy=""
    predication=0
    file_path=""
    img_url=None
    if request.method=="POST":
        user_inp=request.files['file']
        type=request.form.get('uploadType')
        fileType=user_inp.content_type.find('video')
        if "video" in user_inp.content_type:
            file_path=os.path.join(r'D:/Projects/The DeepFake Recognizer/Flask/static/Images/Data/VDS',user_inp.filename)
            predication=user_inp.save(file_path)
        elif "image" in user_inp.content_type:
            file_path=os.path.join(r'D:/Projects/The DeepFake Recognizer/Flask/static/Images/Data/IMG',user_inp.filename)
            user_inp.save(file_path)
            predication=analyzeImg(file_path)
            print("Fake" if predication > 0.5 else "Real")  
            img_url = f"/static/Images/Data/IMG/{user_inp.filename}"

        return render_template('image.html', accuracy=predication, img=img_url)
    elif request.method=="GET":
        return render_template('image.html',accuracy=None, img=None)
# def showImg(path):
#     img=cv.imread(path)
#     img_rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.show()

#     cv.waitKey
# # def detectFake(user_inp,type):
#     # if(type=="video"):
#     #     file=cv.VideoCapture(1)
#     # elif(type=="image"):
#     #     file=cv.imread(user_inp)
#     #     cv.imwrite("D:/Projects/The DeepFake Recognizer/Flask/imgData",file)
#     #     print("Received")  

def analyzeImg(imgPath):
    #img processing
    img=cv.imread(imgPath)
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img=cv.resize(img,(64,64))
    img_arr=img.astype('float32')/255.0
    img_arr=np.expand_dims(img_arr,axis=0)

    #model loading and predication
    model=load_model("D:/Projects/The DeepFake Recognizer/Flask/Model/deepfake_model.keras")

    predict=model.predict(img_arr)

    return predict

    


    
    
if __name__ == "__main__":
    app.run(debug=True)
