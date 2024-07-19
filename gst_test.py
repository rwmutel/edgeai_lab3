import cv2

cap = cv2.VideoCapture('Test.mp4')
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

hostip = '224.0.0.0'  # Multicast IP address
gst_out = ('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast '
           f'! rtph264pay config-interval=1 pt=96 ! udpsink host={hostip} port=5000 auto-multicast=true')
out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (w, h), True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

cap.release()
out.release()
