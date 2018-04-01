import face_recognition
import os
import cv2

FACES_IMAGE_DIR = '/usr/src/app/weigths/data/faces'

class FaceRecognitionModel:

    def __init__(self, *args, **kwargs):
        face_images = os.listdir(FACES_IMAGE_DIR)
        self.face_encodings = []
        self.known_face_names = []
        for i in face_images:
            self.known_face_names.append(i.split('.')[0]) # remove .jpg extension
            img = face_recognition.load_image_file(FACES_IMAGE_DIR + '/' + i)
            self.face_encodings.append(face_recognition.face_encodings(img)[0])

    def recognizeFaces(self, image):
        small_image = cv2.resize(image, (160, 160))
        print("resized" , small_image.shape)
        small_image = small_image[:, :, ::-1]
        face_locations = face_recognition.face_locations(small_image, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(small_image, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            print('matches: ', matches)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append({'name': name })

        return face_names

