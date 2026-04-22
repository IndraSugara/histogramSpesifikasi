import io
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, request, jsonify

from histogram_spec import (
    extract_histogram,
    run_histogram_specification,
    apply_mapping,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  #


def rgb_to_hsv(rgb_array):

    img = rgb_array.astype(np.float64) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # Hue
    H = np.zeros_like(Cmax)
    mask_r = (Cmax == R) & (delta != 0)
    mask_g = (Cmax == G) & (delta != 0)
    mask_b = (Cmax == B) & (delta != 0)
    H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / delta[mask_r])) % 360
    H[mask_g] = (60 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 120)
    H[mask_b] = (60 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 240)
    H = H / 360.0  

    S = np.zeros_like(Cmax)
    nonzero = Cmax != 0
    S[nonzero] = delta[nonzero] / Cmax[nonzero]

    V = Cmax

    return H, S, V


def hsv_to_rgb(H, S, V):

    H360 = H * 360.0
    C = V * S
    X = C * (1 - np.abs((H360 / 60.0) % 2 - 1))
    m = V - C

    R1 = np.zeros_like(H)
    G1 = np.zeros_like(H)
    B1 = np.zeros_like(H)

    idx = (H360 < 60)
    R1[idx], G1[idx], B1[idx] = C[idx], X[idx], 0
    idx = (H360 >= 60) & (H360 < 120)
    R1[idx], G1[idx], B1[idx] = X[idx], C[idx], 0
    idx = (H360 >= 120) & (H360 < 180)
    R1[idx], G1[idx], B1[idx] = 0, C[idx], X[idx]
    idx = (H360 >= 180) & (H360 < 240)
    R1[idx], G1[idx], B1[idx] = 0, X[idx], C[idx]
    idx = (H360 >= 240) & (H360 < 300)
    R1[idx], G1[idx], B1[idx] = X[idx], 0, C[idx]
    idx = (H360 >= 300)
    R1[idx], G1[idx], B1[idx] = C[idx], 0, X[idx]

    R = np.clip((R1 + m) * 255, 0, 255).astype(np.uint8)
    G = np.clip((G1 + m) * 255, 0, 255).astype(np.uint8)
    B = np.clip((B1 + m) * 255, 0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)