import numpy as np

# Colores RGB → clase
CLASS_COLORS = {
    (187, 70, 156): 0,  # Lunar regolith
    (120, 0, 200): 1,   # Impact crater
    (232, 250, 80): 2,  # Rock
    (173, 69, 31): 3,   # Mountain
    (34, 201, 248): 4,  # Sky
}

NUM_CLASSES = len(CLASS_COLORS)

def rgb_to_class(mask_rgb):
    """
    mask_rgb: PIL Image or numpy array (H, W, 3)
    returns: numpy array (H, W) with class indices
    """
    mask_np = np.array(mask_rgb)
    h, w, _ = mask_np.shape

    class_mask = np.zeros((h, w), dtype=np.int64)

    for rgb, class_id in CLASS_COLORS.items():
        matches = np.all(mask_np == rgb, axis=-1)
        class_mask[matches] = class_id

    return class_mask
