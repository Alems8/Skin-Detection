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
import matplotlib.pyplot as plt


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


def getArrays(FrameTitles, MaskTitles):
    width = 320
    height = 240
    t = 0
    matrixFrame = []
    matrixMask = []

    for i in range(0, len(MaskTitles)):
        imgFrame = PIL.Image.open(FrameTitles[i])
        imgFrame = imgFrame.resize((width, height))

        imgMask = PIL.Image.open(MaskTitles[i])
        imgMask = imgMask.resize((width, height))

        for k in range(0, height):
            for j in range(0, width):
                matrixFrame.append([])
                matrixMask.append([])

                rgbFrame = imgFrame.getpixel((j, k))
                hlsFrame = colorsys.rgb_to_hls(rgbFrame[0] / 255, rgbFrame[1] / 255, rgbFrame[2] / 255)

                rgbMask = imgMask.getpixel((j, k))
                hlsMask = colorsys.rgb_to_hls(rgbMask[0] / 255, rgbMask[1] / 255, rgbMask[2] / 255)

                for s in range(0, 3):
                    matrixFrame[t].append(hlsFrame[s])

                if (hlsMask[1] < 0.1):

                    matrixMask[t].append("Black")
                else:

                    matrixMask[t].append("White")
                t += 1
    return np.array(matrixFrame), np.array(matrixMask)


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

    fig = plot_confusion_matrix(clf, X_testscaled, y_test, display_labels=["Black", "White"])
    fig.figure_.suptitle("Confusion Matrix for Skin Detection")
    plt.show()

    return y_show_1, y_show_2


def imageResults(array, p, y):
    img = PIL.Image.new("RGBA", (320, 240), (0, 0, 0))
    k = 0

    for i in range(0, 240):
        for j in range(0, 320):
            if (y[k] == "White"):
                img.putpixel((j, i), (255, 255, 255, 0))
            k += 1

    img1 = PIL.Image.open(array)
    img1 = img1.resize((320, 240))
    img1.paste(img, (0, 0), mask=img)
    img1.show()
    img1.save("D:\AI\ExamProject\Results\Skin_Detection_Result{p}.jpg")


def main():
    FrameTitles, MaskTitles = HaveAlreadyFrames(False)
    InputArray, LabelArray = getArrays(FrameTitles, MaskTitles)

    y1, y2 = compute(InputArray, LabelArray)
    imageResults(FrameTitles[1], 1, y1)
    imageResults(FrameTitles[-1], 2, y2)


if __name__ == "__main__":
    main()