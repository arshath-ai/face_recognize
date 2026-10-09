import cv2, numpy, os  # Import libraries

size = 4  # Not used (can be removed)

# Haar Cascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

datasets = 'datasets'  # Dataset folder

print('Training...')

# Lists to store images, labels, and names
(images, labels, names, id) = ([], [], {}, 0)

# Go through dataset folders (each folder = one person)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir  # Map ID to person name
        subjectpath = os.path.join(datasets, subdir)

        # Read all images of that person
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename

            label = id  # Assign numeric label

            # Read image in grayscale
            images.append(cv2.imread(path, 0))
            labels.append(int(label))

        id += 1  # Move to next person

(width, height) = (200, 200)  # Resize all face images

# Convert lists to numpy arrays
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# Create FisherFace recognizer model
model = cv2.face.FisherFaceRecognizer_create()

# Train model with dataset
model.train(images, labels)

# Load face detector
face_cascade = cv2.CascadeClassifier(haar_file)

# Start webcam
webcam = cv2.VideoCapture(2)

cnt = 0  # Counter for unknown detection

while True:
    (_, im) = webcam.read()  # Capture frame

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Crop face region
        face = gray[y:y + h, x:x + w]

        # Resize face to match training size
        face_resize = cv2.resize(face, (width, height))

        # Predict face
        prediction = model.predict(face_resize)

        # Draw result box
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # If confidence is good (lower value = better match)
        if prediction[1] < 800:
            name = names[prediction[0]]

            cv2.putText(
                im,
                '%s - %.0f' % (name, prediction[1]),
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (51, 255, 255)
            )

            print(name)
            cnt = 0

        else:
            cnt += 1
            cv2.putText(
                im,
                'Unknown',
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0)
            )

            # If unknown detected many times
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("unk.jpg", im)  # Save unknown face image
                cnt = 0

    # Show output window
    cv2.imshow('OpenCV', im)

    # Exit when ESC pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release camera and close windows
webcam.release()
cv2.destroyAllWindows()