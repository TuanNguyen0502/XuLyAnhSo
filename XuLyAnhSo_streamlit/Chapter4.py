import numpy as np
import cv2
L = 256

def FrequencyFiltering(imgin, H):
    # không cần mở rộng ảnh có kích thước PxQ
    f = imgin.astype(np.float32)
    # Bước 1: DFT
    F = np.fft.fft2(f)

    # Bước 2: Shift vào the center of the image
    F = np.fft.fftshift(F)

    # Bước 3: Nhân F với H
    G = F * H

    # Bước 4: Shift return
    G = np.fft.ifftshift(G)

    # Bước 5: IDFT
    g = np.fft.ifft2(G)
    gR = np.clip(g.real, 0, L-1)
    imgout = gR.astype(np.uint8)

    return imgout

def Spectrum(imgin):
    # không cần mở rộng ảnh có kích thước PxQ
    f = imgin.astype(np.float32)
    # Bước 1: DFT
    F = np.fft.fft2(f)

    # Bước 2: Shift vào the center of the image
    F = np.fft.fftshift(F)

    # Bước 3: tính pho
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S, 0, L-1)
    S = S.astype(np.uint8)
    return S

def CreateMoireFilter(M, N):
    # Tạo bộ lọc H là số phức, có phần ảo bằng 0
    H = np.ones((M, N), np.complex64)
    

    u1, v1 = 45, 59
    u2, v2 = 86, 59
    u3, v3 = 39, 119
    u4, v4 = 83, 119

    u5, v5 = 45, 59
    u6, v6 = 86, 59
    u7, v7 = 39, 119
    u8, v8 = 83, 119

    D0 = 10
    for u in range(0, M):
        for v in range(0, N):
            # u1, v1            
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
            # u2, v2            
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
            # u3, v3            
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
            # u4, v4            
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
            # u5, v5            
            Duv = np.sqrt((u-u5)**2 + (v-v5)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0 
            # u6, v6
            Duv = np.sqrt((u-u6)**2 + (v-v6)**2)         
            if Duv <= D0:
                H.real[u,v] = 0.0
            # u7, v7
            Duv = np.sqrt((u-u7)**2 + (v-v7)**2)            
            if Duv <= D0:
                H.real[u,v] = 0.0
            # u8, v8
            Duv = np.sqrt((u-u8)**2 + (v-v8)**2)            
            if Duv <= D0:
                H.real[u,v] = 0.0

    return H


def CreateButterworthNotchRejectFilter(P, Q):
    # Tạo bộ lọc H là số phức, có phần ảo bằng 0
    H = np.ones((P,Q,2), np.float32)
    H[:,:,1] = 0.0

    u1, v1 = 45, 59
    u2, v2 = 86, 59
    u3, v3 = 39, 119
    u4, v4 = 83, 119
    D0 = 10
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            r = 1.0
            # u1, v1            
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # u2, v2            
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # u3, v3            
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # u4, v4            
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # đối xứng của u1, v1            
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # đối xứng của u2, v2            
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # đối xứng của u3, v3            
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            # đối xứng của u4, v4            
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv <= D0:
                if Duv >= 1e-10:
                    r = r * 1.0/(1.0 + np.power(D0/Duv, n))
                else:
                    r = 0.0

            H[u,v,0] = r
    return H

def CreateVerticalNotchRejecFilter(P, Q):
    # Tạo bộ lọc H là số phức, có phần ảo bằng 0
    H = np.ones((P,Q,2), np.float32)
    H[:,:,1] = 0.0
    D0 = 7
    D1 = 7
    for u in range(0, P):
        for v in range(0, Q):
            if not u in range(Q//2-D1, Q//2+D1): 
                D = v-Q//2
                if abs(D) <= D0:
                    H[u,v,0] = 0
    return H

def RemoveMoireSimple(imgin):
    M, N = imgin.shape
    H = CreateMoireFilter(imgin, M, N)
    imgout = FrequencyFiltering(imgin, H) 
    return imgout

def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateButterworthNotchRejectFilter(P, Q)
    imgout = FrequencyFiltering(imgin, H) 
    return imgout

def RemoveInterference(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateVerticalNotchRejecFilter(P, Q)
    imgout = FrequencyFiltering(imgin, H) 
    return imgout