import streamlit as st
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import face_recognition as fr
import os
import matplotlib.pyplot as plt
from deepface import DeepFace



header= st.container()
models= st.container()
model1= st.container()
model2= st.container()
model3= st.container()

path = "new_data/train/"
# 
known_names = []
known_name_encodings = []
# 
images = os.listdir(path)
# 
for _ in images:
     image = fr.load_image_file(path + _)
     image_path = path + _
                     #print(image_path)
     encoding = fr.face_encodings(image)[0]
# 
     known_name_encodings.append(encoding)
     known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

#for first part(befor apply button)
with header:
    img = Image.open("welcome.jpeg")
    st.image(img, width=700)
    st.title("Welcom to Facial Exptessions and Reconstruction app!")
    st.header("Are you ready to have a unique a experience with us ?")
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        try:
            ret, frame = camera.read() ## read one image from a video
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #FRAME_WINDOW.image(frame)
            result = DeepFace.analyze(frame, actions = ['emotion'])
                    #print (faceCascade. empty())
            faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            faces = faceCascade.detectMultiScale(frame,1.1,4)
                    # Draw a rectanale around the faces
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            font= cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame,
                    result['dominant_emotion'],
                    (50,50),
                    font,2,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        # Use putText) method for
        # inserting text on video
            FRAME_WINDOW.image(frame)
        except:
            pass
    else:
        st.write('Stopped')
     #take imge
    uploaded_file = st.file_uploader("1.Uploade a pictute")
     #pick one    
    model_choose = st.radio(
     "2.Choose the action you want us to apply to your picture:",
     ('Identity Recognition', 'Emotion Recognition', "3D People's Image Reconstruction"))

#for second part(aftr apply button)
with models:
    if st.button('Apply'):
      if uploaded_file is  None:
          st.error("you should uplode picture and choose one action befor prassing apply button!")              
      else:
          if model_choose == 'Identity Recognition':
               with model1:
                   
                    st.title(model_choose)
                    placeholder = st.empty()
                    placeholder.text('Please wait...')
                    test_image= fr.load_image_file('new_data/test/'+uploaded_file.name)
                    face_locations = fr.face_locations(test_image)
                    face_encodings = fr.face_encodings(test_image, face_locations)
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                       matches = fr.compare_faces(known_name_encodings, face_encoding)
                       name = ""
# 
                       face_distances = fr.face_distance(known_name_encodings, face_encoding)
                       best_match = np.argmin(face_distances)
# 
                       if matches[best_match]:
                           name = known_names[best_match]
# 
                       cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
                       cv2.rectangle(test_image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
# 
                       font = cv2.FONT_HERSHEY_DUPLEX
                       cv2.putText(test_image, name, (left + 6, bottom - 6), font, 0.9, (255, 255, 255), 1)
                    result_image=test_image
                    result_text= name
                    st.image(result_image, width=450)
          if model_choose == 'Emotion Recognition':
              with model2:
                    st.title(model_choose)
                    placeholder = st.empty()
                    placeholder.text('Please wait...')
                    test_image= fr.load_image_file('new_data/test/'+uploaded_file.name)
                    prediction= DeepFace.analyze(test_image)

                    faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

                    gray= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

                    faces= faceCascade.detectMultiScale(gray, 1.1, 4)

                    for (x,y,w,h) in faces:
                        cv2.rectangle(test_image, (x,y), (x+w, y+h), (0,255,0), 2)

                    font= cv2.FONT_HERSHEY_SIMPLEX

                    cv2.putText(test_image,
                                str(prediction['dominant_emotion']+' '+prediction['gender']),
                                (50,50),
                                font,2,
                                (0,0,255),
                                2,
                                cv2.LINE_4)

                    cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                    result_text=str(prediction['dominant_emotion']+' '+prediction['gender'])
                    st.image(test_image, width=450)
          if model_choose == "3D People's Image Reconstruction":
              with model3:
                    st.title(model_choose)
                    placeholder = st.empty()
                    placeholder.text('Coming soon...')
                   # output pathes
                    #obj_path = 'result_%s_256.obj' % file_name
                    #out_img_path = 'result_%s_256.png' % file_name
                    #video_path = 'result_%s_256.mp4' % file_name
                    #video_display_path = 'result_%s_256_display.mp4' % file_name
                    #result_video= ("https://www.youtube.com/watch?v=KmzA0PiQs6M")
                    #result_video= open("Sample.mp4","rb") 
                    #result_video=result_video.read()
                    #result_text="???"
                    #st.video(result_video)
                    #st.subheader('The 3D reconstruction of the person in the picture:')
                    #st.write(result_text)
    else:
        pass

                

         