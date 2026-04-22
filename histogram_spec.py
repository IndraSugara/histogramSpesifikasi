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
    

def compute_mapping(source_table, target_table, L):
    sk_values = source_table['sk']
    vk_values = target_table['vk']
    
    mapping = {}
    
    for rk in range(L):
        sk = sk_values[rk]
        
        min_diff = float('inf')
        best_zk = 0
        
        for zk in range(L):
            vk = vk_values[zk]
            diff = abs(sk - vk)
            
            if diff < min_diff:
                min_diff = diff
                best_zk = zk
        mapping[rk] = best_zk
    
    return mapping


def apply_mapping(image_array, mapping):
    image = np.array(image_array, dtype=int)
    result = np.zeros_like(image)
    
    for rk, zk in mapping.items():
        result[image == rk] = zk
    
    return result


def calculate_result_histogram(mapping, source_table, L):
    result_hist = [0] * L
    
    for rk, zk in mapping.items():
        result_hist[zk] += int(source_table['frekuensi'][rk])
    
    return result_hist


def run_histogram_specification(nk, pz, L, n):
    source_table = calculate_source_table(nk, L, n)
    
    target_table = calculate_target_table(pz, L)
    
    mapping = compute_mapping(source_table, target_table, L)

    result_histogram = calculate_result_histogram(mapping, source_table, L)
    
    return {
        'source_table': source_table,
        'target_table': target_table,
        'mapping': mapping,
        'result_histogram': result_histogram,
        'L': L,
        'n': n,
    }
