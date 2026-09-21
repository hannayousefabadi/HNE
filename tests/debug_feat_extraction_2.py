import json
from hne.core.paths import PATIENTS, TIF_MAP
from hne.core.data_io import get_s3_loader

pid = 'CH_L_275a'
paths = PATIENTS[pid]
filename = TIF_MAP.get(paths.clean_id)
print(f'TIF_MAP points to: {filename}')
print(f'Raw image prefix: {paths.raw_image_prefix}')

loader = get_s3_loader()
try:
    scale_json = loader.read_json(f'{paths.visium_info}/scalefactors_json.json')
    print('\nScale factors JSON:')
    print(json.dumps(scale_json, indent=2))
except Exception as e:
    print('Could not load scalefactors_json:', e)