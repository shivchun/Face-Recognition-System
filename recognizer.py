import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("classifier.yml")
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (id == 1):
            id = "shiv"
        if (id == 2):
            id = "Arjun"
        if (id == 3):
            id = "Raushan"
        if (id == 4) :
            id = "Prakash"
        #else:
         #   cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red
         #   cv2.putText(img, "Unknown", (x, y+h), font,4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(id), (x, y+h), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()