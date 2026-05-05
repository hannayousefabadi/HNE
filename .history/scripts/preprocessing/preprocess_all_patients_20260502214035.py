from hne.paths import PATIENTS
from hne.data_io import load_visium, load_spots, load_scale_factor, load_he_image


for pid, paths in PATIENTS.items():
    adata = load_visium(paths)
    spots = load_spots(paths)
    scale = load_scale_factor(paths)
    img = load_he_image(paths)




