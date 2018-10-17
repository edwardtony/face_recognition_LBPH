import numpy as np
import cv2
import os
from random import shuffle
from tkinter import *

class UI:

    def __init__(self, master):
        self.master = master
        self.master.geometry('200x200+560+280')
        self.master.title('Welcome')

        self.face_cascade = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')
        self.IMG_SIZE  = 64

        self.start_components()

    def start_components(self):
        self.class_name = StringVar()

        # self.nombreImagen.set("adsas")
        Button(self.master, text="Run", bg="#E4E4E4", bd=1, command=self.run).grid(padx=55, pady=10)
        # Button(self.master, text="Train", bg="#E4E4E4", bd=1, command=self.train).grid(padx=55, pady=5)
        Button(self.master, text="Add Class", bg="#E4E4E4", bd=1, command=self.add_class).grid(padx=55, pady=5)
        Entry(self.master, textvariable=self.class_name, width=10).grid(padx=10, pady= 10)

    def get_data(self, path):
        faces = []
        labels = []
        classes = {}
        for file_name in os.listdir(path):
            if file_name == ".DS_Store" : continue

            classes[file_name.split('_')[3]] = file_name.split('_')[2]
            data = np.load('{}/{}'.format(path, file_name))
            faces = [*faces, *data[0]]
            labels = [*labels, *data[1]]

        shuffle(faces*5)
        shuffle(labels*5)
        print(classes)
        return faces, labels, classes


    def run(self):
        if len(os.listdir('dataset')) in range(2): return

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,360)
        self.cap.set(4,240)
        padding = 10

        faces, labels, classes = self.get_data('dataset')
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))

        while True:
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:

                cv2.rectangle(gray,(x + padding,y + padding),(x+w - padding,y+h - padding),(0,0,0),2)
                roi_gray = gray[y + padding :y+h - padding, x + padding:x+w - padding]
                roi_gray = np.array(cv2.resize(roi_gray, (self.IMG_SIZE,self.IMG_SIZE))).reshape(self.IMG_SIZE, self.IMG_SIZE,1)

                label= face_recognizer.predict(roi_gray)
                label_class = classes[str(label[0])] if label[1] < 100 else 'Unknown'
                print(label)
                cv2.putText(gray,'{} - {}'.format(label_class, label[1]),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,255,255),1,cv2.LINE_AA)

            cv2.imshow('original',gray)
            cv2.moveWindow('original', 400,250)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                break


    def add_class(self):
        if ( self.class_name.get() == '' ) : return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,360)
        self.cap.set(4,240)

        class_name = self.class_name.get()
        label = len(os.listdir('dataset'))
        training_data_X = []
        training_data_Y = []
        roi_gray = None
        padding = 10
        i = 0
        data_lenght = 20

        while True and i < data_lenght:
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:

                # cv2.rectangle(gray,(x + padding,y + padding),(x+w - padding,y+h - padding),(0,0,0),2)
                roi_gray = gray[y + padding :y+h - padding, x + padding:x+w - padding]
                roi_gray = np.array(cv2.resize(roi_gray, (self.IMG_SIZE,self.IMG_SIZE))).reshape(self.IMG_SIZE, self.IMG_SIZE,1)
                training_data_X.append(roi_gray)
                training_data_Y.append(label)

                # label = "Feliz " if self.cnn.model.predict([roi_gray])[0][0] >= 0.5 else "Neutral"

                cv2.imshow('face',roi_gray)
                cv2.moveWindow('face', 400,250)
                i = i + 1

            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break

        np.save('dataset/train_data_{}_{}_{}.npy'.format(class_name.upper(), label, data_lenght), [training_data_X, training_data_Y])
        print("DATA LENGHT: ", len(training_data_X))
        self.cap.release()
        cv2.destroyAllWindows()

def main():

    root = Tk()
    ui = UI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
