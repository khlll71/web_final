from ultralytics import YOLO
import streamlit as st
import tempfile
import PIL
import cv2
import os

import settings
from collections import Counter


counter = Counter()

def load_model(model_path):
    model = YOLO(model_path)
    return model


def _display_detected_frames(model, st_frame, image):

    #image = cv2.resize(image, (720, int(720*(9/16))))
    res = model.predict(image,device=1,retina_masks=True,augment=True,visualize=True,save=False)
    res_plotted = res[0].plot(conf=False,font_size=20,font='Ariel.ttf',pil=True)
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_rtsp_stream(model):
    source_rtsp = st.sidebar.text_input("RTMP stream URL")

    col1, col2 = st.columns(2)
    with col1:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        upload_shit = st.image(default_image_path,use_column_width=True)
    if st.sidebar.button('Start Detect'):
        upload_shit.empty()
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                    
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTMP stream: " + str(e))



def play_stored_video(model):

    source_vid = st.sidebar.file_uploader("Choose video...", type=('AVI', 'mp4','webm'))
    col1, col2 = st.columns(2)
    
    with col1:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        upload_shit = st.image(default_image_path,use_column_width=True)
        if source_vid is not None:
            upload_shit.empty()
            st.video(source_vid)
    with col2:
        if st.sidebar.button('Start Detect Video'):
            
            if source_vid:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, source_vid.name)
                with open(path, "wb") as f:
                    f.write(source_vid.getvalue())
                try:
                    vid_cap = cv2.VideoCapture(path)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(model,
                                                    st_frame,
                                                    image
                                                    )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.sidebar.error("Error loading video: " + str(e))



 
