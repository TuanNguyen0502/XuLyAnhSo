import cv2
import numpy as np

L = 256

def ConnectedComponents(imgin):
    nguong = 200
    _, temp = cv2.threshold(imgin, nguong, L - 1, cv2.THRESH_BINARY)
    imgout = cv2.medianBlur(temp, 7)

    n, label = cv2.connectedComponents(imgout)
    a = np.zeros(n, np.uint32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                a[r] = a[r] + 1
    s = 'Co %d thanh phan lien thong' % (n - 1)
    cv2.putText(imgout, s, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    for r in range(1, n):
        s = '%3d %5d' % (r, a[r])
        cv2.putText(imgout, s, (10, 20 * (r + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    return imgout

def RemoveSmallRice(imgin):
    # 81 là kích thước lớn nhất của hạt gạo
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    nguong = 100
    _, temp = cv2.threshold(temp, nguong, L - 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    n, label = cv2.connectedComponents(temp)
    a = np.zeros(n, np.uint32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                a[r] = a[r] + 1
    max_value = np.max(a)
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0: 
                if a[r] > 0.7 * max_value:
                    imgout[x, y] = L - 1
    return imgout