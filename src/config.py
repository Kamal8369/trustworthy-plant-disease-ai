from pathlib import Path

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PLANTVILLAGE_TRAIN_DIR = Path("PlantVillageDataset/train_val_test/train")
PLANTVILLAGE_VAL_DIR = Path("PlantVillageDataset/train_val_test/val")
PLANTDOC_ROOT = Path(".")

# Maps PlantVillage class names to the corresponding PlantDoc folder names.
PLANTDOC_CLASS_MAPPING = {
    "Tomato__Tomato_mosaic_virus": "Tomato_leaf_mosaic_virus",
    "Tomato_Late_blight": "Tomato_leaf_late_blight",
    "Tomato_Early_blight": "Tomato_Early_blight_leaf",
    "Tomato_Leaf_Mold": "Tomato_mold_leaf",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Potato___Late_blight": "Potato_leaf_late_blight",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato_leaf_yellow_virus",
    "Potato___Early_blight": "Potato_leaf_early_blight",
    "Tomato_healthy": "Tomato_leaf",
    "Tomato_Bacterial_spot": "Tomato_leaf_bacterial_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato_two_spotted_spider_mites_leaf",
    "Pepper__bell___healthy": "Bell_pepper_leaf",
    "Pepper__bell___Bacterial_spot": "Bell_pepper_leaf_spot",
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
