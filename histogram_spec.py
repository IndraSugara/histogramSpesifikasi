import numpy as np


def extract_histogram(image_array, L=256):
    image = np.array(image_array, dtype=int).flatten()
    nk = np.zeros(L, dtype=int)
    
    for pixel in image:
        if 0 <= pixel < L:
            nk[pixel] += 1
    
    return nk.tolist()


def calculate_source_table(nk, L, n):
    nk = np.array(nk, dtype=np.float64)

    pdf = nk / n

    cdf = np.cumsum(pdf)

    sk = np.round(cdf * (L - 1)).astype(int)
    
    return {
        'intensitas': list(range(L)),
        'frekuensi': nk.tolist(),
        'pdf': pdf.tolist(),
        'cdf': cdf.tolist(),
        'sk': sk.tolist(),
    }


def calculate_target_table(pz, L):
    pz = np.array(pz, dtype=np.float64)
    
    cdf = np.cumsum(pz)
    
    vk = np.round(cdf * (L - 1)).astype(int)
    
    return {
        'intensitas': list(range(L)),
        'pdf': pz.tolist(),
        'cdf': cdf.tolist(),
        'vk': vk.tolist(),
    }