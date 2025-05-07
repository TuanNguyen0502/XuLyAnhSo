import cv2
import numpy as np
import Chapter4 as c4
L = 256

def CreateMotionFilter(P, Q):
    f = open('result.txt', 'w')
    # Tạo bộ lọc H là số phức
    H = np.zeros((P,Q,2), np.float32)
    a = 0.1
    b = 0.1
    T = 1.0
    dem = 0
    s = ''
    for u in range(0, P):
        for v in range(0, Q):
            phi = np.pi*((u-P//2)*a + (v-Q//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = T
                IM = 0.0
                s = 'dem: %4d, u: %4d, V: %4d' % (dem, u, v) + '\n'
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H[u,v,0] = RE
            H[u,v,1] = IM    
    f.write(s)
    f.close()
    return H

def CreateInverseMotionFilter(P, Q):
    # Tạo bộ lọc H là số phức
    H = np.zeros((P,Q,2), np.float32)
    a = 0.1
    b = 0.1
    T = 1.0
    for u in range(0, P):
        for v in range(0, Q):
            phi = np.pi*((u-P//2)*a + (v-Q//2)*b)
            temp = np.sin(phi)
            if np.abs(temp) < 1.0e-6:
                RE = np.cos(phi)/T
                IM = np.sin(phi)/T
            else:
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/T
            H[u,v,0] = RE
            H[u,v,1] = IM
    return H

def CreateInverseMotionWeinerFilter(P, Q):
    # Tạo bộ lọc H là số phức
    H = np.zeros((P,Q,2), np.float32)
    a = 0.1
    b = 0.1
    T = 1.0
    for u in range(0, P):
        for v in range(0, Q):
            phi = np.pi*((u-P//2)*a + (v-Q//2)*b)
            temp = np.sin(phi)
            if np.abs(temp) < 1.0e-6:
                RE = np.cos(phi) / T
                IM = np.sin(phi) / T
            else:
                RE = phi / (T * np.sin(phi)) * np.cos(phi)
                IM = phi / T
            H[u, v, 0] = RE
            H[u, v, 1] = IM

    S = np.sqrt(H[u, v, 0]**2 + H[u, v, 1]**2)
    K = 0.0001
    HeSo = S / (S + K)
    H[u, v, 0] = H[u, v, 0] * HeSo
    H[u, v, 1] = H[u, v, 1] * HeSo 
    return H

def CreateMotion(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateMotionFilter(P, Q)
    imgout = c4.FrequencyFiltering(imgin, H) 
    return imgout

def DeMotion(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateInverseMotionFilter(P, Q)
    imgout = c4.FrequencyFiltering(imgin, H) 
    return imgout

def DeMotionWeiner(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateInverseMotionWeinerFilter(P, Q)
    imgout = c4.FrequencyFiltering(imgin, H) 
    return imgout