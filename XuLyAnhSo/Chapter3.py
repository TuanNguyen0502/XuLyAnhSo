import cv2
import numpy as np

L = 256

def Negative(imgin):
    # M: độ cao của ảnh, N: độ rộng của ảnh
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8) #+ np.uint8(L)
    # Quét ảnh
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = np.uint8(s)
    return imgout

def NegativeColor(imgin):
    # M: độ cao của ảnh, N: độ rộng của ảnh, C: số kênh màu
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), dtype=np.uint8) #+ np.uint8(L)
    # Quét ảnh
    for x in range(0, M):
        for y in range(0, N):
            # Ảnh của opencv là ảnh BGR
            # Ảnh của pillow là ảnh RGB - pillow là module ảnh của python
            b = imgin[x, y, 0]
            b = L - 1 - b

            g = imgin[x, y, 1]
            g = L - 1 - g

            r = imgin[x, y, 2]
            r = L - 1 - r

            imgout[x, y, 0] = np.uint8(b)
            imgout[x, y, 1] = np.uint8(g)
            imgout[x, y, 2] = np.uint8(r)
    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8)
    c = (L - 1) / np.log(1.0 * L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c * np.log(1.0 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

def PowerLaw(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8)
    gamma = 5.0
    c = np.power(L - 1.0, 1.0 - gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r == 1
            s = c * np.power(1.0 * r, gamma)
            imgout[x, y] = np.uint8(s)
    return imgout

def PiecewiseLine(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8)
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            # Doan I
            if r < r1:
                s = 1.0 * s1/r1*r
            # Doan II
            elif r < r2:
                s = 1.0 * (s2 - s1)/(r2 - r1)*(r - r1) + s1
            # Doan III
            else:
                s = 1.0 * (L - 1 - s2)/(L - 1 - r2)*(r - r2) + s2
            imgout[x, y] = np.uint8(s)
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, L, 3), dtype=np.uint8) + np.uint8(255)
    hist = np.zeros((L), dtype=np.int32)
    # Tính histogram
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            hist[r] = hist[r] + 1
    p = 1.0 * hist / (M * N)
    # Vẽ histogram
    scale = 3000
    for r in range(0, L):
        cv2.line(imgout, (r, M - 1), (r, M - 1 - np.int32(scale * p[r])), (255, 0, 0))
    return imgout

def HistogramEqualization(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8)
    hist = np.zeros((L), dtype=np.int32)
    # Tính histogram
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            hist[r] = hist[r] + 1
    p = 1.0 * hist / (M * N)
    s = np.zeros((L), dtype=np.float64)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    
    return imgout

def HistogramEqualizationColor(imgin):
    b = imgin[:, :, 0]
    g = imgin[:, :, 1]
    r = imgin[:, :, 2]

    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    
    imgout = imgin.copy()
    imgout[:, :, 0] = b
    imgout[:, :, 1] = g
    imgout[:, :, 2] = r
    return imgout

def LocalHistogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a:x + a + 1, y - b:y + b + 1]
            w = cv2.equalizeHist(w)
            imgout[x, y] = w[a, b]
    return imgout

def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), dtype=np.uint8)
    mean, stddev = cv2.meanStdDev(imgin)
    mG = mean[0, 0]
    sigmaG = stddev[0, 0]
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a:x + a + 1, y - b:y + b + 1]
            mean, stddev = cv2.meanStdDev(w)
            msxy = mean[0, 0]
            sigmasxy = stddev[0, 0]
            if (k0 * mG <= msxy <= k1 * mG) and (k2 * sigmaG <= sigmasxy <= k3 * sigmaG):
                imgout[x, y] = np.uint8(C * imgin[x, y])
            else:
                imgout[x, y] = imgin[x, y]
    return imgout

def SmoothBox(imgin):
    m = 21
    n = 21
    w = np.zeros((m, n), dtype=np.float32) + np.float32(1.0 / (m * n))
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def Sharp(imgin):
    w = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = imgin - laplacian
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
    # Tính toán gradient của ảnh
    sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    
    grad_x = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    grad_y = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    imgout = abs(grad_x) + abs(grad_y)
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)

    return imgout