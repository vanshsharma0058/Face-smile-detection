import cv2
import os
import time
from datetime import datetime
import glob
import pygame

# Initialize pygame mixer for audio
pygame.mixer.init()
click_sound = pygame.mixer.Sound("C:/Users/owner2/Desktop/python/face_detection/face_detect_project/Audio/sound.wav")  # 🎵 your sound file here
smile_sound = pygame.mixer.Sound("C:/Users/owner2/Desktop/python/face_detection/face_detect_project/Audio/happy.wav")
# In this project we draw green rectangle,blue rectangle,red rectangle around the face ,eye and smile 
# Load the pre-trained face detection model (Haar Cascade)
face_detector = cv2.CascadeClassifier(
    "C:/Users/owner2/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

eye_detector = cv2.CascadeClassifier("C:/Users/owner2/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_eye.xml")

#Check if cascade loaded correctly
if eye_detector.empty():
       print("❌ Error loading eye cascade")

smile_dectector = cv2.CascadeClassifier("C:/Users/owner2/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_smile.xml")

# Create folder for saving smile images
if not os.path.exists("screenshots"):
    os.makedirs("screenshots",exist_ok=True)
if not os.path.exists("smiles"):
    os.makedirs("smiles",exist_ok=True)

# Start capturing video from the default webcam (0)
video_capture = cv2.VideoCapture(0)

# To avoid saving multiple smile screenshots repeatedly
last_smile_time = 0
smile_delay = 3  # seconds

while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()
    if not ret or frame is None:
       print("⚠️ Failed to grab frame from camera!")
       break

    # Convert the frame to grayscale (face detection works better in gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.06,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    total_eyes=0
    total_smiles=0
    smile_detected_now=False
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Region of Interest (ROI) for detecting the eyes and smile inside the face
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

    # detect the eye inside the face ROI
        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2,minSize=(10,10))
        total_eyes+=len(eyes)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    
    # detect the smile inside face ROI
        smiles = smile_dectector.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=30)
        total_smiles+=len(smiles)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
            smile_detected_now=True
            #Save the smile face image
            smile_img=frame[y:y+h,x:x+w]
            
            #for saving the screenshot of smile face only
            #filename=f"smiles/smile_{int(time.time())}.jpg"
            #cv2.imwrite(filename,smile_img)
            #print(f"✅ Smile saved: {filename}")
   
    #Show counter
    cv2.putText(frame,f"Faces:{len(faces)}",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
    cv2.putText(frame,f"Eyes:{total_eyes}",(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2)
    cv2.putText(frame,f"Smiles:{total_smiles}",(10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)
    
    #Real-time clock
    now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame,f"Time:{now}",(450,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(255,255,255),2)
    
    # Show the frame with rectangles on a window
    cv2.imshow("Live Face Detection with eye and smile", frame)
    
     
     # If smile is detected and enough time passed, save full screenshot
     # Audio feedback
    current_time = time.time()
    if smile_detected_now and (current_time - last_smile_time) > smile_delay:
        smile_sound.play() # Play smile sound
        smile_file = f"smiles/smile_frame_{int(current_time)}.jpg"
        cv2.imwrite(smile_file, frame)
        print(f"😄 Smile detected! Screenshot saved: {smile_file}")
        last_smile_time = current_time

    # Break the loop if 'a' key is pressed
    key =cv2.waitKey(1) & 0xFF
    if key== ord("a"):
        break
    elif key == ord("s"):
    # Save screenshot in the screenshots folder
       snap_file = f"screenshots/screenshot_{int(time.time())}.jpg"
       cv2.imwrite(snap_file, frame)
       print(f"📸 Screenshot saved: {snap_file}")
       
       click_sound.play()
    # Auto-delete if more than 5 screenshots
       screenshot_files = sorted(glob.glob("screenshots/*.jpg"), key=os.path.getctime)
       if len(screenshot_files) > 5:
          oldest = screenshot_files[0]
          os.remove(oldest)
          print(f"🗑️ Deleted old screenshot: {oldest}")
 
# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
