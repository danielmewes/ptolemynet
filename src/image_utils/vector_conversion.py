def img_to_vecs(img):
    output = np.zeros((x_size * y_size, 2))
    rescaled_img = 2.0 * img.flatten() - 1.0
    output[:, 0] = np.sin(rescaled_img * np.pi / 2.0)
    output[:, 1] = np.cos(rescaled_img * np.pi / 2.0)
    return output

def vecs_to_img(vecs):
    v = vecs.reshape((x_size, y_size, 2))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    img = np.arcsin(v[:, :, 0:1]) / np.pi * 2.0
    img = (1.0 + img) / 2.0
    img = 255.0 * np.clip(img, 0.0, 1.0)
    return img
