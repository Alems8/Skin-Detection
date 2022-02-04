# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import math
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import PIL

from moviepy.editor import VideoFileClip
import os

# d1={}
# key = [["b"],["n"]]
arr = np.array("label")

Titles = ["D:/AI/ExamProject/Video/1.avi", "D:/AI/ExamProject/Video/1m.avi"]
FrameTitles = []
MaskTitles = []
duration = [math.inf, math.inf]
for i in range(0, len(Titles)):
    videoclip = VideoFileClip(Titles[i])
    duration[i % 2] = videoclip.duration
    for currentdur in np.arange(0, duration[duration[0] > duration[1]]):
        print("Evviva")
        frame_filename = os.path.join("D:/AI/ExamProject/Frame", f"{i}.{currentdur}.jpg")
        videoclip.save_frame(frame_filename, currentdur)
        if (i % 2 == 0):
            FrameTitles.append(frame_filename)
        else:
            MaskTitles.append(frame_filename)

for i in range(0, 2000):
    a = random.random()
    if (a < 0.3):
        # arr=np.vstack(arr,key[0])
        arr = np.append(arr, "n")
    else:
        # arr=np.vstack([arr,key[1]])
        arr = np.append(arr, "b")
arr = np.delete(arr, 0)
np.vstack(arr)

arr2 = np.array(0)
for i in range(0, 2000):
    h = random.random()
    l = random.random()
    s = random.random()
    arr2 = np.append(arr2, [h, l, s])
arr2 = np.delete(arr2, 0)
arr2 = arr2.reshape(2000, 3)

df = pd.DataFrame({"label": arr, "h": arr2[:, 0], "l": arr2[:, 1], "s": arr2[:, 2]})
y = df["label"]
x = df.drop("label", axis=1)
x_test_pred = x.tail(1000)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_testscaled = sc_X.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict(X_testscaled)
y_pred = clf.predict(x_test_pred)
print(clf.score(X_testscaled, y_test))

fig = plot_confusion_matrix(clf, X_testscaled, y_test, display_labels=["bianco", "nero"])
fig.figure_.suptitle("Test")
plt.show()

img = PIL.Image.new("RGBA", (320, 240), (0, 0, 0))
k = 0

for i in range(0, 50):
    for j in range(0, 20):
        if (y_pred[k] == "b"):
            img.putpixel((i, j), (255, 255, 255, 0))  # bianco

        k += 1

img2 = PIL.Image.open(FrameTitles[1])
img2.paste(img, (0, 0), mask=img)
img2.show()
img2.save("D:\AI\ExamProject\Results\Skin_Detection_Result.jpg")
