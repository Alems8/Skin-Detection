# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:30:35 2022

@author: aleal
"""

from moviepy.editor import VideoFileClip
import math
import os
import numpy as np
import PIL
import colorsys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd


def HaveAlreadyFrames(answer, Framepath=None, MaskPath=None):
    if (answer == False):
        Titles = ["D:/AI/ExamProject/Video/1.avi", "D:/AI/ExamProject/Video/1m.avi",
                  "D:/AI/ExamProject/Video/2.avi", "D:/AI/ExamProject/Video/2m.avi",
                  "D:/AI/ExamProject/Video/3.avi", "D:/AI/ExamProject/Video/3m.avi",
                  "D:/AI/ExamProject/Video/4.avi", "D:/AI/ExamProject/Video/4m.avi",
                  "D:/AI/ExamProject/Video/5.avi", "D:/AI/ExamProject/Video/5m.avi",
                  "D:/AI/ExamProject/Video/6.avi", "D:/AI/ExamProject/Video/6m.avi",
                  "D:/AI/ExamProject/Video/7.avi", "D:/AI/ExamProject/Video/7m.avi",
                  "D:/AI/ExamProject/Video/8.avi", "D:/AI/ExamProject/Video/8m.avi",
                  "D:/AI/ExamProject/Video/9.avi", "D:/AI/ExamProject/Video/9m.avi",
                  "D:/AI/ExamProject/Video/10.avi", "D:/AI/ExamProject/Video/10m.avi",
                  "D:/AI/ExamProject/Video/11.avi", "D:/AI/ExamProject/Video/11m.avi",
                  "D:/AI/ExamProject/Video/12.avi", "D:/AI/ExamProject/Video/12m.avi"]

        FrameTitles = []
        MaskTitles = []
        duration = [math.inf, math.inf]
        for i in range(0, len(Titles)):
            videoclip = VideoFileClip(Titles[i])
            duration[i % 2] = videoclip.duration
            if (duration[0] <= duration[1]):
                num_frames = duration[0]
            else:
                num_frames = duration[1]

            for currentdur in np.arange(0, num_frames):
                frame_filename = os.path.join("D:/AI/ExamProject/Frame", f"{i}.{currentdur}.jpg")
                videoclip.save_frame(frame_filename, currentdur)
                if (i % 2 == 0):
                    FrameTitles.append(frame_filename)
                else:
                    MaskTitles.append(frame_filename)
                    duration = [math.inf, math.inf]

    else:
        FrameTitles = Framepath
        MaskTitles = MaskPath

    return FrameTitles, MaskTitles


def getArrays(titles):
    width = 320
    height = 240
    t = 0
    matrix = []
    for i in range(0, len(titles)):
        img = PIL.Image.open(titles[i])
        img = img.resize((width, height))

        for k in range(0, height):
            for j in range(0, width):
                matrix.append([])
                rgb = img.getpixel((j, k))
                hls = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
                if (isinstance(titles[0], float)):
                    for s in range(0, 3):
                        matrix[t].append(hls[s])
                else:
                    if (hls[1] < 0.1):

                        matrix[t].append("Black")
                    else:

                        matrix[t].append("White")
                t += 1
    return matrix


def compute(InputArray, LabelArray):
    df = pd.DataFrame({"label": LabelArray[:, 0], "h": InputArray[:, 0], "l": InputArray[:, 1], "s": InputArray[:, 2]})
    y = df["label"]
    x = df.drop("label", axis=1)
    x_show_1 = x.head(153600)
    x_show_1 = x_show_1.tail(76800)
    x_show_2 = x.tail(76800)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    sc_X = StandardScaler()

    X_trainscaled = sc_X.fit_transform(x_train)
    X_testscaled = sc_X.transform(x_test)

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_trainscaled, y_train)

    clf.predict(X_testscaled)

    y_show_1 = clf.predict(x_show_1)
    y_show_2 = clf.predict(x_show_2)

    print(clf.score(X_testscaled, y_test))

    return y_show_1, y_show_2


def imageResults(array, i, y):
    img = PIL.Image.new("RGBA", (320, 240), (0, 0, 0))
    k = 0

    for i in range(0, 240):
        for j in range(0, 320):
            if (y[k] == "White"):
                img.putpixel((j, i), (255, 255, 255, 0))  # bianco
    k += 1

    img1 = PIL.Image.open(array[0])
    img1 = img1.resize((320, 240))
    img1.paste(img, (0, 0), mask=img)
    img1.show()
    img1.save("D:\AI\ExamProject\Results\Skin_Detection_Result{i}.jpg")


def main():
    FrameTitles, MaskTitles = HaveAlreadyFrames(False)
    InputArray = np.array(getArrays(FrameTitles))
    LabelArray = np.array(getArrays(MaskTitles))
    y1, y2 = compute(InputArray, LabelArray)
    imageResults(FrameTitles, 1, y1)
    imageResults(FrameTitles, 2, y2)


if __name__ == "__main__":
    main()