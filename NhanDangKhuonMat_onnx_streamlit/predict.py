import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os

def main():
    st.subheader('Nhận dạng khuôn mặt')
    FRAME_WINDOW = st.image([])
    cap = cv.VideoCapture(0)

    # Load models relative to this file's directory
    current_dir = os.path.dirname(__file__)
    detector_path = os.path.join(current_dir, 'face_detection_yunet_2023mar.onnx')
    recognizer_path = os.path.join(current_dir, 'face_recognition_sface_2021dec.onnx')
    stop_img_path = os.path.join(current_dir, 'stop.jpg')
    svc_path = os.path.join(current_dir, 'svc.pkl')

    if 'stop' not in st.session_state:
        st.session_state.stop = False
        stop = False

    press = st.button('Stop')
    if press:
        st.session_state.stop = not st.session_state.stop
        if st.session_state.stop:
            cap.release()

    print('Trang thai nhan Stop', st.session_state.stop)

    if 'frame_stop' not in st.session_state:
        frame_stop = cv.imread(stop_img_path)
        st.session_state.frame_stop = frame_stop
        print('Đã load stop.jpg')

    if st.session_state.stop == True:
        FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
        return

    svc = joblib.load(svc_path)
    mydict = ['HoangDuyen', 'MinhVuong', 'PhiHiep', 'Tuan']

    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

                # Ensure coordinates are within image bounds
                x1, y1, w, h = coords[0], coords[1], coords[2], coords[3]
                x2, y2 = x1 + w, y1 + h
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(input.shape[1], x2), min(input.shape[0], y2)

                # Extract face embedding
                face_img = input[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                face_embedding = recognizer.feature(face_img)

                # Flatten the face embedding
                face_embedding_flat = face_embedding.flatten()

                # Predict identity
                name_idx = svc.predict([face_embedding_flat])[0]
                name = mydict[name_idx]

                # Add name label
                cv.putText(input, name, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    detector = cv.FaceDetectorYN.create(
        detector_path,
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(recognizer_path, "")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv.putText(frame,result,(1,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
