# face_recog.py
from person_db import Person
from person_db import Face
from person_db import PersonDB
import person_db
import face_recognition
import numpy as np
from datetime import datetime
from datetime import timedelta
import os
import cv2 
import threading
import time
import traceback


class Observer():
    # called when Face recognizer is started
    def on_start(self, fr):
        pass

    # called when a person appears for the first time
    def on_new_person(self, person):
        pass

    # called when the person appears again
    def on_person(self, person):
        pass

    # called when Face recognizer is stopped
    def on_stop(self, fr):
        pass

class Observable():
    def __init__(self):
        self.observers = []

    def register_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    # notify that face recognizer is started
    def notify_start(self):
        for observer in self.observers:
            observer.on_start(self)

    # notify that a person appears first
    def notify_new_person(self, person):
        for observer in self.observers:
            observer.on_new_person(person)

    # notify that the person appears again
    def notify_person(self, person):
        for observer in self.observers:
            observer.on_person(person)

    # notify that face recognizer is stopped
    def notify_stop(self):
        for observer in self.observers:
            observer.on_stop(self)

class FaceRecog(Observable):
    def __init__(self,person_db, settings):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.person_name = ""
        self.value = 0
        self.ret = []
        self.process_this_frame = True
        self.settings = settings
        self.running = False
        self.pdb = person_db
        Observable.__init__(self)


        src_file = self.settings.srcfile
        self.cam = cv2.VideoCapture(src_file)
        self.resrc = self.cam
        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = '/Users/johnsmac/Downloads/train_face/'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)




    def get_frame(self, ret, frame):
        # Grab a single frame of video

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        i = 0
        person = Person()
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            person.add_face(self.face_locations)
            now = datetime.now()
            str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)
                self.value = min_value
                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = 'unknown'
                if min_value < self.settings.threshold:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]            
                    self.person_name = name
                    #print(self.person_name)
                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,0), 5)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (100, 125, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #print(name, "{:.3f}".format((1-self.value)*100)+"%")

            if name != 'unknown':
                cv2.putText(frame, name +' '+ str("{:.4f}".format((1-self.value)*100)) + '%', (left + 6, bottom - 6), font, 1.0, (0,0, 255), 1)
            else: cv2.putText(frame, name , (left + 6, bottom - 6), font, 1.0, (0,0, 255), 1)


        return frame    

    def start_running(self):
        if self.running == True:
            print('already running')
            return
        src = self.cam
        if not src.isOpened():
            self.error_string = "cannot open inputfile"
            return -1
            
        frame_width = src.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = src.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = src.get(5)
        self.frames_to_skip = int(round(self.frame_rate * self.settings.sbf))

        s = "* srcfile = " + str(self.settings.srcfile)
        if self.settings.srcfile == 0:
            s += ' (webcam)'
        s += "\n* size = %dx%d" % (src.get(3), src.get(4))
        ratio = self.settings.resize_ratio
        s += "\n* resize_ratio = " + str(ratio)
        s += " -> %dx%d" % (int(src.get(3) * ratio), int(src.get(4) * ratio))
        s += "\n* frame_rate = %.3f f/s" % self.frame_rate
        s += "\n* process every " + str(self.frames_to_skip) + " frames"
        self.source_info_string = s

        self.src = src
        self.running = True
        t = threading.Thread(target=self.run)
        t.start()

    def stop_running(self):
        self.running = False
        self.status_string = 'Face recognizer is not running.'

    def run(self):
        self.notify_start()
        frame_id = 0
        i = 0
        processing_time = 0
        while self.running:
            try:
                ret, frame = self.src.read()
                if frame is None:
                    break
                frame_id += 1
                if frame_id % self.frames_to_skip != 0:
                    continue
                start_time = time.time()
                processing_time += time.time() - start_time
                i += 1
                self.last_frame = frame
                d = self.get_frame(ret, frame)
                self.process_frame(frame)

                dt = timedelta(seconds=int(frame_id/self.frame_rate))
                s = 'Face recognizer running time: ' + str(dt) + '.'
                s += '\nTotal ' + str(i) + ' frames are processed.'
                s += '\nAverage processing time per frame is %.3f seconds.' % (processing_time / i)
                self.status_string = s
            except:
                traceback.print_exc()
                print('break loop...')
                break

        self.src.release()
        self.stop_running()
        self.pdb.save_db()
        self.pdb.print_persons()
        self.notify_stop()
        
    def get_jpg_bytes(self):
        ret, frame = self.resrc.read()
        frame = self.get_frame(ret, frame)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

    def locate_faces(self, frame):
        #start_time = time.time()
        ratio = self.settings.resize_ratio
        if ratio == 1.0:
            rgb = frame[:, :, ::]
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
            rgb = small_frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb)
        #elapsed_time = time.time() - start_time
        #print("locate_faces takes %.3f seconds" % elapsed_time)
        if ratio == 1.0:
            return boxes
        boxes_org_size = []
        for box in boxes:
            (top, right, bottom, left) = box
            left = int(left / ratio)
            right = int(right / ratio)
            top = int(top / ratio)
            bottom = int(bottom / ratio)
            box_org_size = (top, right, bottom, left)
            boxes_org_size.append(box_org_size)
        return boxes_org_size

    def get_face_image(self, frame, box):
        img_height, img_width = frame.shape[:2]
        (box_top, box_right, box_bottom, box_left) = box
        box_width = box_right - box_left
        box_height = box_bottom - box_top
        crop_top = max(box_top - box_height, 0)
        pad_top = -min(box_top - box_height, 0)
        crop_bottom = min(box_bottom + box_height, img_height - 1)
        pad_bottom = max(box_bottom + box_height - img_height, 0)
        crop_left = max(box_left - box_width, 0)
        pad_left = -min(box_left - box_width, 0)
        crop_right = min(box_right + box_width, img_width - 1)
        pad_right = max(box_right + box_width - img_width, 0)
        face_image = frame[:,:,::]
        if (pad_top == 0 and pad_bottom == 0):
            if (pad_left == 0 and pad_right == 0):
                return face_image
        #print(face_image.shape)
        padded = cv2.copyMakeBorder(face_image,  pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
        return padded

    def detect_faces(self, frame):
        boxes = self.locate_faces(frame)
        if len(boxes) == 0:
            return []
        # faces found
        faces = []
        now = datetime.now()
        str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'
        encodings = face_recognition.face_encodings(frame, boxes)
        for i, box in enumerate(self.face_locations):
            face_image = self.get_face_image(frame, box)
            face = Face(str_ms + str(i) + ".jpg", face_image, encodings[i])
            face.location = box
            faces.append(face)
        return faces

    def compare_with_known_persons(self, face, persons):
        if len(persons) == 0:
            return None
        # see if the face is a match for the faces of known person
        encodings = [person.encoding for person in persons]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.settings.threshold and self.person_name != "unknown":
            # face of known person
            persons[index].add_face(face)
            # re-calculate encoding
            persons[index].calculate_average_encoding()
            face.name = persons[index].name
            return persons[index]

    def compare_with_unknown_faces(self, face, unknown_faces):
        if len(unknown_faces) == 0:
            # this is the first face
            unknown_faces.append(face)
            face.name = "unknown"
            self.person_name = face.name
            return
        encodings = [face.encoding for face in unknown_faces]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.settings.threshold:
            # two faces are similar - create new person with two faces
            person = Person()
            newly_known_face = unknown_faces.pop(index)
            person.add_face(newly_known_face)
            person.add_face(face)
            person.calculate_average_encoding()
            person.name = self.person_name 
            face.name = person.name
            newly_known_face.name = person.name
            return person
        else:
            # unknown face
            unknown_faces.append(face)
            face.name = "unknown"
            self.person_name = face.name
            return None


    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        for face in faces:
            person = self.compare_with_known_persons(face, self.pdb.persons)
            if person:
                self.notify_person(person)
                person.update_last_face_time()
                continue
            person = self.compare_with_unknown_faces(face, self.pdb.unknown.faces)
            if person:
                self.pdb.persons.append(person)
                self.notify_new_person(person)
                person.update_last_face_time()

        # draw names
        for face in faces:
            ret, frame = self.src.read()
            d = self.get_frame(ret, frame)

if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    ret, frame = self.resrc.read()
    while True:
        frame = face_recog.get_frame(ret, frame)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
