from flask import Flask, render_template, Response
import cv2
from matplotlib.pyplot import text
import numpy as np
from keras.models import model_from_json
from datetime import datetime
import time
import os,random
import subprocess
  

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")


face_haar_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# global text

app = Flask(__name__)

camera = cv2.VideoCapture(0)
now = time.time()###For calculate seconds of video
future = now + 20

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1, 0)
        # frame = cv2.resize(frame, (460, 360))
        if not ret:
            break
        else:
            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
                text=emotion_dict[maxindex]
            # print(text)
            resized_img = cv2.resize(frame, (1000, 700))  
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            # key = cv2.waitKey(30)& 0xff
        # if time.time() > future:##after 20second music will play
        #     try:
        #         cv2.destroyAllWindows()
                # mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                # if text == 'Angry':
                #     randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Melody_songs/"))
                #     print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                #     file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Melody_songs/' + randomfile)
                #     subprocess.call([mp, file])

                # if text == 'Happy':
                #     randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/party_songs/"))
                #     print('You are smiling :) ,I playing special song for you: ' + randomfile)
                #     file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/party_songs/' + randomfile)
                #     subprocess.call([mp, file])

                # if text == 'Fearful':
                #     randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Motivational_songs/"))
                #     print('You have fear of something ,I playing song for you: ' + randomfile)
                #     file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Motivational_songs/' + randomfile)
                #     subprocess.call([mp, file])

                # if text == 'Sad':
                #     randomfile = random.choice(os.listdir("C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Happy_songs/"))
                #     print('You are sad,dont worry:) ,I playing song for you: ' + randomfile)
                #     file = ('C:/Users/konrupes/Documents/GitHub/Emotion_detection_with_CNN/songs/Happy_songs/' + randomfile)
                #     subprocess.call([mp, file])
            #     break
            # except :
            #     print('Please stay focus in Camera frame atleast 15 seconds & run again this program:)')
            #     break

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
