import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep

# prende la cartella faccie e la codifica in modo che il pc lo riconosca


def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

# stessa cosa di prima


def unknown_image_encoded(img):

    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

# apre una finestra con l'immagine (test.jpg) e crea una lista delle faccie che riconosce quelle che non riconosce sono sconosciute


def classify_face(im):

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(
        img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # se non trova faccie che riconosce ci scrive sconosciuto
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Sconosciuto"

        # Cerca nelle immagini la foto più somigliante
        face_distances = face_recognition.face_distance(
            faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Disenga un rettangolo interno alla faccia
            cv2.rectangle(img, (left-20, top-20),
                          (right+20, bottom+20), (255, 255, 0), 2)

            # Scrive una label sotto alla foto
            cv2.rectangle(img, (left-20, bottom - 15),
                          (right+20, bottom+20), (0, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left - 20, bottom + 15),
                        font, 1.0, (255, 255, 255), 2)

    # Se c'è corrispondenza mostra il nome sull immagine
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names


print(classify_face("test.jpg"))
