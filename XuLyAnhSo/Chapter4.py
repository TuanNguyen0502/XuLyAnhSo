import numpy as np
import cv2
L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    # Bước 1 và 2: Tạo ảnh có kích thước PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = 1.0*imgin / (L-1)

    # Bước 3: Nhân fp với (-1)^(x+y)
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4: Tính biến đổi Fourier thuận DFT
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)
    FR = F[:,:,0]
    FI = F[:,:,1]
    S = np.sqrt(FR**2 + FI**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8)
    return imgout


def CreateNotchFilter(P, Q):
    # Tạo bộ lọc H là số phức, có phần ảo bằng 0
    H = np.ones((P, Q, 2), np.complex64)
    H[:,:,1] = 0.0

    u1, v1 = 45, 58
    u2, v2 = 86, 58
    u3, v3 = 41, 119
    u4, v4 = 83, 119

    u5, v5 = P - u1, Q - v1
    u6, v6 = P - u2, Q - v2
    u7, v7 = P - u3, Q - v3
    u8, v8 = P - u4, Q - v4

    D0 = 15
    for u in range(0, P):
        for v in range(0, Q):
            # u1, v1
            Duv = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u2, v2
            Duv = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u3, v3
            Duv = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u4, v4
            Duv = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u5, v5
            Duv = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u6, v6
            Duv = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u7, v7
            Duv = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # u8, v8
            Duv = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
    return H

def DrawNotchFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateNotchFilter(P, Q)
    HR = H[:,:,0] * (L-1)
    imgout = HR.astype(np.uint8)
    return imgout

def DrawNotchPeriodFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateNotchPeriodFilter(P, Q)
    HR = H[:,:,0] * (L-1)
    imgout = HR.astype(np.uint8)
    return imgout

def CreateNotchPeriodFilter(P, Q):
    H = np.ones((P, Q, 2), np.float32)
    H[:,:,1] = 0.0
    D0 = 10
    v0 = Q // 2
    for u in range(0, P):
        for v in range(0, Q):
            if u not in range(Q//2-10, Q//2+10+1): 
                if abs(v - v0) <= D0:
                    H[u,v,0] = 0.0
    return H

def RemoveNotchSimple(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)    
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = 1.0 * imgin

    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    H = CreateNotchFilter(P, Q)
    H_real = H[:, :, 0].astype(np.float32)  # Use only the real part of H

    # Expand H_real to match the shape of F
    H_expanded = np.zeros_like(F)
    H_expanded[:, :, 0] = H_real
    H_expanded[:, :, 1] = 0  # Imaginary part is zero

    G = cv2.mulSpectrums(F, H_expanded, flags=cv2.DFT_ROWS)

    g = cv2.idft(G, flags=cv2.DFT_SCALE)

    gR = g[:M, :N, 0]
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]
    gR = np.clip(gR, 0, L - 1)

    imgout = gR.astype(np.uint8)
    return imgout

def RemovePeriodNoise(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)    
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = 1.0 * imgin

    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    H = CreateNotchPeriodFilter(P, Q)
    H_real = H[:, :, 0].astype(np.float32)  # Use only the real part of H

    # Expand H_real to match the shape of F
    H_expanded = np.zeros_like(F)
    H_expanded[:, :, 0] = H_real
    H_expanded[:, :, 1] = 0  # Imaginary part is zero

    G = cv2.mulSpectrums(F, H_expanded, flags=cv2.DFT_ROWS)

    g = cv2.idft(G, flags=cv2.DFT_SCALE)

    gR = g[:M, :N, 0]
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]
    gR = np.clip(gR, 0, L - 1)

    imgout = gR.astype(np.uint8)
    return imgout