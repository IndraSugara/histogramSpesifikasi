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


def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def numpy_to_base64_gray(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode='L')
    return pil_to_base64(img)


def numpy_to_base64_rgb(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    return pil_to_base64(img)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#fff', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_histogram_charts(src_nk, tgt_nk, res_hist, L, label='V'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(L)
    bw = 1.0

    for ax, data, color, title in [
        (axes[0], src_nk,  '#5B8DEE', f'Histogram Sumber ({label})'),
        (axes[1], tgt_nk,  '#FF6B6B', f'Histogram Target ({label})'),
        (axes[2], res_hist, '#51CF66', f'Histogram Hasil ({label})'),
    ]:
        ax.bar(x, data, width=bw, color=color, edgecolor=color, linewidth=0.3)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Intensitas', fontsize=9)
        ax.set_ylabel('Frekuensi', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2.0)
    return fig_to_base64(fig)


def generate_comparison(src_img, tgt_img, res_img, is_color):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    cmap = None if is_color else 'gray'
    kwargs = {} if is_color else {'vmin': 0, 'vmax': 255}

    for ax, img, title in [
        (axes[0], src_img, 'Citra Sumber'),
        (axes[1], tgt_img, 'Citra Target'),
        (axes[2], res_img, 'Citra Hasil'),
    ]:
        ax.imshow(img, cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(pad=2.0)
    return fig_to_base64(fig)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate_image', methods=['POST'])
def calculate_image():
    try:
        if 'source_image' not in request.files or request.files['source_image'].filename == '':
            return jsonify({'error': 'Upload citra sumber terlebih dahulu.'}), 400
        if 'target_image' not in request.files or request.files['target_image'].filename == '':
            return jsonify({'error': 'Upload citra target terlebih dahulu.'}), 400

        mode = request.form.get('mode', 'grayscale')

        src_pil = Image.open(request.files['source_image'].stream).convert('RGB')
        tgt_pil = Image.open(request.files['target_image'].stream).convert('RGB')

        src_rgb = np.array(src_pil, dtype=np.uint8)
        tgt_rgb = np.array(tgt_pil, dtype=np.uint8)

        L = 256
        src_h, src_w = src_rgb.shape[:2]
        tgt_h, tgt_w = tgt_rgb.shape[:2]
        src_n = src_h * src_w
        tgt_n = tgt_h * tgt_w

        if mode == 'color':
            src_H, src_S, src_V = rgb_to_hsv(src_rgb)
            tgt_H, tgt_S, tgt_V = rgb_to_hsv(tgt_rgb)

            src_v8 = np.clip(src_V * 255, 0, 255).astype(np.uint8)
            tgt_v8 = np.clip(tgt_V * 255, 0, 255).astype(np.uint8)

            src_nk = extract_histogram(src_v8, L)
            tgt_nk = extract_histogram(tgt_v8, L)
            pz = [freq / tgt_n for freq in tgt_nk]


            result = run_histogram_specification(src_nk, pz, L, src_n)

            res_v8 = apply_mapping(src_v8, result['mapping'])
            res_V = res_v8.astype(np.float64) / 255.0

            res_rgb = hsv_to_rgb(src_H, src_S, res_V)

            chart_b64 = generate_histogram_charts(src_nk, tgt_nk, result['result_histogram'], L, label='V channel')
            comp_b64 = generate_comparison(src_rgb, tgt_rgb, res_rgb, is_color=True)
            res_img_b64 = numpy_to_base64_rgb(res_rgb)

            display_src = src_rgb
            display_tgt = tgt_rgb

        else:
            src_gray = np.array(src_pil.convert('L'), dtype=np.uint8)
            tgt_gray = np.array(tgt_pil.convert('L'), dtype=np.uint8)

            src_nk = extract_histogram(src_gray, L)
            tgt_nk = extract_histogram(tgt_gray, L)
            pz = [freq / tgt_n for freq in tgt_nk]

            result = run_histogram_specification(src_nk, pz, L, src_n)
            res_gray = apply_mapping(src_gray, result['mapping'])

            chart_b64 = generate_histogram_charts(src_nk, tgt_nk, result['result_histogram'], L, label='Grayscale')
            comp_b64 = generate_comparison(src_gray, tgt_gray, res_gray, is_color=False)
            res_img_b64 = numpy_to_base64_gray(res_gray)

        mapping_list = [{'rk': rk, 'zk': result['mapping'][rk]} for rk in range(L)]

        return jsonify({
            'source_table': result['source_table'],
            'target_table': result['target_table'],
            'mapping': mapping_list,
            'result_histogram': result['result_histogram'],
            'target_nk': tgt_nk,
            'chart': chart_b64,
            'comparison': comp_b64,
            'result_image': res_img_b64,
            'src_size': {'w': src_w, 'h': src_h},
            'tgt_size': {'w': tgt_w, 'h': tgt_h},
            'L': L,
            'src_n': src_n,
            'tgt_n': tgt_n,
            'mode': mode,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(debug=False, host='0.0.0.0', port=port)
