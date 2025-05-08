import cv2
import numpy as np

L = 256

def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    imgout = cv2.erode(imgin, w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgout = cv2.dilate(imgin, w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp
    return imgout

def Contour(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    n = len(contour)
    for i in range(0, n - 1):
        x1 = contour[i, 0, 0]
        y1 = contour[i, 0, 1]
        x2 = contour[i + 1, 0, 0]
        y2 = contour[i + 1, 0, 1]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)

    x1 = contour[n-1, 0, 0]
    y1 = contour[n-1, 0, 1]
    x2 = contour[0, 0, 0]
    y2 = contour[0, 0, 1]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return imgout

def ConvexHull(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    
    hull = cv2.convexHull(contour)
    n = len(hull)
    for i in range(0, n - 1):
        x1 = hull[i, 0, 0]
        y1 = hull[i, 0, 1]
        x2 = hull[i + 1, 0, 0]
        y2 = hull[i + 1, 0, 1]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)

    x1 = hull[n-1, 0, 0]
    y1 = hull[n-1, 0, 1]
    x2 = hull[0, 0, 0]
    y2 = hull[0, 0, 1]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return imgout

def DefectDetect(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    p = cv2.convexHull(contour, returnPoints=False)
    n = len(p)
    for i in range(0, n-1):
        vi_tri_1 = p[i, 0]
        vi_tri_2 = p[i + 1, 0]
        x1 = contour[vi_tri_1, 0, 0]
        y1 = contour[vi_tri_1, 0, 1]
        x2 = contour[vi_tri_2, 0, 0]
        y2 = contour[vi_tri_2, 0, 1]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)

    vi_tri_1 = p[n - 1, 0]
    vi_tri_2 = p[0, 0]
    x1 = contour[vi_tri_1, 0, 0]
    y1 = contour[vi_tri_1, 0, 1]
    x2 = contour[vi_tri_2, 0, 0]
    y2 = contour[vi_tri_2, 0, 1]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)

    defects = cv2.convexityDefects(contour, p)
    nguong_do_sau = np.max(defects[:, :, 3]) // 2
    n = len(defects)
    for i in range(0, n):
        do_sau = defects[i, 0, 3]
        if do_sau > nguong_do_sau:
            vi_tri_khuyet = defects[i, 0, 2]
            x = contour[vi_tri_khuyet, 0, 0]
            y = contour[vi_tri_khuyet, 0, 1]
            cv2.circle(imgout, (x, y), 5, (0, 255, 0), -1)
    return imgout

def HoleFill(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(imgout, None, (104, 295), (0, 0, 255))
    return imgout

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