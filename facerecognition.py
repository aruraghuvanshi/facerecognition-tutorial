import warnings

warnings.filterwarnings('ignore')
import face_recognition as fr
import cv2
import numpy as np
import pickle
import time
import glob
import os

DATASETPATH = 'C:\\Users\\void_\\dataset'
IMAGEPATH = 'C:\\Users\\void_\\test\\md.jpg'
verbose = True
MODE = 'image'  # 'webcam', 'image' and 'video'
TRAINFLAG = False


def scale(path):
    im = cv2.imread(path)
    if im.shape[0] > 2000:
        scale = 0.25  # large jpegs
    elif 1000 > im.shape[0] >= 2000:
        scle = 0.5
    elif 500 > im.shape[0] >= 1000:
        scale = 0.75  # whatsapp viber etc
    else:
        scale = 1
    return scale


if MODE == 'image':
    scale = scale(path=IMAGEPATH)

TRAIN_NAMES = ['Aru', 'Charlize Theron', 'PM Modi']


def trainAndEncode(path=DATASETPATH, verbose=verbose):
    face_encodings = []
    allfiles = glob.glob(path + '\\*.jpg')  # to read through the dataset
    if verbose: print('Commencing Model Training...')
    time.sleep(1)

    for x in allfiles:

        if verbose: print(f'Training Model on {x}')
        image = fr.load_image_file(os.path.abspath(x))
        image_face_encoding = fr.face_encodings(image)[0]
        face_encodings.append(image_face_encoding)

    if verbose: print(f'Face Training Successful')
    return face_encodings


def runFaceRecognizer(path=DATASETPATH, verbose=verbose, mode=MODE):
    if TRAINFLAG:
        face_train = trainAndEncode(path)
        with open('face_encodings.txt', 'wb') as fp:
            pickle.dump(face_train, fp)
    else:
        with open('face_encodings.txt', 'rb') as fo:
            face_train = pickle.load(fo)

    face_locations = []
    face_names = []
    process = True
    time.sleep(1)

    if mode == 'webcam':

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if process:
                face_locations = fr.face_locations(frame)
                face_encodings = fr.face_encodings(frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = fr.compare_faces(face_train, face_encoding, tolerance=0.6)
                    names = 'Unrecognized'
                    face_distances = fr.face_distance(face_train, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        names = TRAIN_NAMES[best_match_index]
                    face_names.append(names)

            process = not process
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 12), font, 0.7, (0, 255, 0), 1)

            cv2.imshow('Frame', frame)
            k = cv2.waitKey(30)
            if k == 27:
                break
        cap.release()
        if verbose: print('Program Ended')


    elif mode == 'image':

        frame = cv2.imread(IMAGEPATH)
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = small_frame[:, :, ::-1]
        frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

        if process:
            face_locations = fr.face_locations(frame)
            face_encodings = fr.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(face_train, face_encoding, tolerance=0.6)
                names = 'Unrecognized'
                face_distances = fr.face_distance(face_train, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    names = TRAIN_NAMES[best_match_index]
                face_names.append(names)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 12), font, 0.7, (0, 255, 0), 1)

        cv2.imshow('Frame', frame)
        cv2.waitKey(0)
        if verbose: print('Program Ended')

    cv2.destroyAllWindows()
    return
    

runFaceRecognizer()
