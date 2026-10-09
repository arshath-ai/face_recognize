import cv2, os  # Import OpenCV and OS modules    

# Haar Cascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

datasets = 'datasets'      # Main folder to store dataset
sub_data = 'dinas'         # Person name folder (label)

# Create path for saving images
path = os.path.join(datasets, sub_data)

# Create folder if it doesn't exist
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (200, 200)  # Resize face images to 200x200

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(haar_file)

# Start webcam (0 = default camera, 1 = external camera)
webcam = cv2.VideoCapture(0)

count = 1  # Image counter

while count < 31:  # Capture 30 images
    print(count)

    (_, im) = webcam.read()  # Read frame from webcam

    # Convert image to grayscale (better for face detection)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)   

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop face region
        face = gray[y:y+h, x:x+w]

        # Resize face image
        face_resize = cv2.resize(face, (width, height))

        # Save face image in dataset folder
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)

    count += 1  # Increase image count

    # Show webcam window
    cv2.imshow('OpenCV', im)

    # Wait for key press (27 = ESC)
    key = cv2.waitKey(10)

    if key == 27:
        break  # Exit if ESC is pressed

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()