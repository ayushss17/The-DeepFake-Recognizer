from flask import Flask, render_template, flash, request, url_for, redirect, session
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import shutil
from io import BytesIO
from PIL import Image
import base64
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def home(): 
    return render_template("index.html")
@app.route('/image',methods=['GET','POST'])
@app.route('/image#result',methods=['GET','POST'])
def image():
    accuracy=""
    predict=0
    file_path=""
    img_url=None
    if request.method=="POST":
        user_inp=request.files['file']
        type=request.form.get('uploadType')
        fileType=user_inp.content_type.find('video')
        if "video" in user_inp.content_type:
            file_exists=False
            dir="D:/Projects/The DeepFake Recognizer/Flask/static/Images/Data/VDS"
            finalPath=""

            for paths in os.listdir(dir):
                subFold=os.path.join(dir,paths)
                if os.path.isdir(subFold):
                    filePath=os.path.join(subFold,user_inp.filename)

                    if os.path.exists(filePath):
                        print("File already exists")
                        finalPath=filePath
                        file_exists=True
                        break
            
            if not file_exists:
                saveFolder=os.path.join(dir,os.listdir(dir)[0])
                savePath=os.path.join(saveFolder,user_inp.filename)
                user_inp.save(savePath)
                print("File saved")
                finalPath=savePath

            finalPath=os.path.normpath(finalPath)

            predict,label,frames=analyzeVideo(finalPath)
            filetype=user_inp.content_type.split('/')
            return render_template('image.html', accuracy=predict,type=user_inp.content_type,frames=frames)


        elif "image" in user_inp.content_type:
            file_exists = False
            label=""
            dir = "D:/Projects/The DeepFake Recognizer/Flask/static/Images/Data/IMG"
            finalPath = "" 
            
            for paths in os.listdir(dir):
                subFold = os.path.join(dir, paths)
                if os.path.isdir(subFold):
                    filePath = os.path.join(subFold, user_inp.filename)
            
                    if os.path.exists(filePath):
                        print("File already exists")
                        finalPath = filePath  
                        file_exists = True
                        break
                    
            if not file_exists:
                saveFolder = os.path.join(dir, os.listdir(dir)[0])
                savePath = os.path.join(saveFolder, user_inp.filename)
                user_inp.save(savePath)
                print("File saved")
                finalPath = savePath
            
            finalPath = os.path.normpath(finalPath)  # Normalize path
            print(finalPath)
            predict, label = analyzeImg(finalPath)
            img_url = f"/static/Images/Data/IMG/{label}/{user_inp.filename}"
            return render_template('image.html', accuracy=predict, img=img_url,type=user_inp.content_type)
        print("Fake" if predict > 0.5 else "Real")  

        
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

def extractFrames(videoPath,frames=30):
    video=cv.VideoCapture(videoPath)
    totalFrames=int(video.get(cv.CAP_PROP_FRAME_COUNT))

    frameIndx=np.linspace(0,totalFrames-1,frames,dtype=int)
    extractedFrames=[]
    originalFrame=[]

    for idx in frameIndx:
        video.set(cv.CAP_PROP_POS_FRAMES,idx)
        ret,frame=video.read()
        originalFrame.append(frame)
        if not ret:
            break
        frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        frame=cv.resize(frame,(64,64))
        frame=frame.astype("float32")/255.0
        extractedFrames.append(frame)

    video.release()

    #padding
    while len(extractedFrames)<frames:
        extractedFrames.append(np.zeros(64,64,3))
    
    return np.array(extractedFrames),np.array(originalFrame)

def frameConversion(frames):
    encodedFrames=[]
    for frame in frames:
            frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            frame=np.clip(frame, 0, 255).astype(np.uint8)
            frame=cv.resize(frame,(200,200) )
            pilImg=Image.fromarray(frame)

            buffer=BytesIO()
            pilImg.save(buffer,format="JPEG")
            img_str=base64.b64encode(buffer.getvalue()).decode('utf-8')
            encodedFrames.append(img_str)

    return encodedFrames

def analyzeVideo(videoPath):
    folder="D:/Projects/The DeepFake Recognizer/Flask/static/Images/Data/VDS"
    model=load_model("D:/Projects/The DeepFake Recognizer/Flask/Model/deepfake_model_video1.keras")
    extFrames,orgFrames=extractFrames(videoPath,frames=30)
    encodedFrames=frameConversion(orgFrames)
    inpBatch=np.expand_dims(extFrames,axis=0)
    predict=model.predict(inpBatch)

    if predict>0.5:
        label="Fake"
        print(label)
        print(predict)
    else:    
        label="Real"
        print(label)
        print(predict)
    catPath=os.path.join(folder,label)
    # shutil.copy(videoPath,catPath)
    return predict,label,encodedFrames
def analyzeImg(imgPath):
    folder="D:/Projects/The DeepFake Recognizer/Flask/static/Images/Data/IMG"
    #img processing
    face_casc=cv.CascadeClassifier('D:/Projects/The DeepFake Recognizer/Flask/Model/haars_face.xml')
    img=cv.imread(imgPath)
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    #Detecting Faces -
    faces=face_casc.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    if(len(faces)==0):
        print("No faces are there")

    x,y,w,h=faces[0]
    img=img[y:y+h,x:x+w]
    img=cv.resize(img,(64,64))
    img_arr=img.astype('float32')/255.0
    img_arr=np.expand_dims(img_arr,axis=0)

    #model loading and predict
    model=load_model("D:/Projects/The DeepFake Recognizer/Flask/Model/deepfake_model_updated1.keras")

    predict=model.predict(img_arr)
    if predict>0.5:
        label="Fake"
        print(label)
        print(predict)
    else:    
        label="Real"
        print(label)
        print(predict)
    catPath=os.path.join(folder,label)
    # shutil.copy(imgPath,catPath)

    return predict,label

    


    
    
if __name__ == "__main__":
    app.run(debug=True)
