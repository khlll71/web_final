from pathlib import Path
import PIL
from collections import Counter

# External packages
import streamlit as st

from ultralytics import YOLO
import settings
import helper
from recognition_records import display_recognition_records , save

st.set_page_config(
    page_title="test",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("""
<style>
h1 {
    background-color: #FF90BC; 
    padding: 20px;
    border-radius: 10px;
    color: #111; 
    font-size: 40px;
    font-weight: bold;
    width: 100%; 
    height: 80px; 
    font-family : Courier, monospace;
}
p {
    font-family : Courier, monospace;
    text-align:center;
    color : #111;
    font-size : 26px;
    font-weight:10;
}
img {
    border-radius: 1.5em;
    padding-top:10px;
}
.stButton > button:first-child { 
    background-color: #8ACDD7;
    color: #fff;
    border: 5px gainsboro;
    border-radius: 20px;
    padding: 10px 20px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    width: 240px;
    text-decoration: none; 
    text-align: center; 
    white-space: normal; 
}
.stButton > button:hover { 
    background-color: #F9F9E0; 
}
.stButton > button:active{
    background-color: #999; 
}
.stDataFrame{
    border-radius : 30px;
    padding-top:20px;
    color:#999; 
    text-align:left !important;
}        
[data-testid=stSidebar] {
    background-color: #F9F9E0;
    font-size:16px;
}

[data-testid=stAppViewContainer] {
    background-color: #FFC0D9;
}
[data-testid=stHeader] {
    background-color: #FFC0D9;
}
video{
    border-radius:30px;
    padding : 20px;
}
</style>
""", unsafe_allow_html=True)



st.title("Object Detection System")

model_type = 'Detection'


if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if 'selected_button' not in st.session_state:
    st.session_state.selected_button = None


image_bt = st.sidebar.button('Select Image')
if image_bt:
    st.session_state.selected_button = "image_bt"
video_bt = st.sidebar.button('Select video')
if video_bt:
    st.session_state.selected_button = "video_bt"
rtmp_bt = st.sidebar.button('Input RTMP')
if rtmp_bt:
    st.session_state.selected_button = "rtmp_bt"
his_bt = st.sidebar.button('View History')
if his_bt:
    st.session_state.selected_button = "his_bt"

source_img = None
if st.session_state.selected_button == "image_bt":
    source_img = st.sidebar.file_uploader(
        "Choose image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path,use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",use_column_width=True)
                
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if st.sidebar.button('Start Detect',key='detect'):
            res = model.predict(uploaded_image,device='cpu',retina_masks=True,save=False)
            boxes = res[0].boxes
            res_plotted = res[0].plot(conf=False)[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                    use_column_width=True)
            tensor_data = boxes.cls
            tensor_data_cpu = tensor_data.cpu()
            numpy_array = tensor_data_cpu.numpy().astype(int)
            counter = Counter(numpy_array)
            names = [ 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife', 'spoon','bowl', 'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
            result_string = "; ".join([f"{names[num]}: {count}" for num, count in counter.items() if num > 0 and count >= 1])
            save(result_string)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No image is uploaded yet!")
    pass
    

elif st.session_state.selected_button == "video_bt":
    helper.play_stored_video(model)
    pass

elif st.session_state.selected_button == "rtmp_bt":
    helper.play_rtsp_stream(model)
    pass


elif st.session_state.selected_button == "his_bt":
    display_recognition_records()
    pass

#else:
#    st.error("Something wrong!")
