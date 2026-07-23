from hne.feature_extraction.phikon_v2 import PhikonExtractor
from hne.core.paths import TILES, PHIKON_FEATURES

def extract_features():
    phikon = PhikonExtractor()
    phikon.extract_patient(patients_dir=TILES, output_dir=PHIKON_FEATURES)

if __name__ == "__main__":
    extract_features()

