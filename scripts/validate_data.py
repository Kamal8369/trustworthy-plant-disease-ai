import torch

from src.config import (
    BATCH_SIZE,
    PLANTDOC_ROOT,
    PLANTVILLAGE_TRAIN_DIR,
    PLANTVILLAGE_VAL_DIR,
    SEED,
)
from src.data import (
    build_dataloaders,
    build_plantdoc_datasets,
    build_plantvillage_datasets,
    class_counts_from_imagefolder,
    class_counts_from_plantdoc,
)
from src.reproducibility import set_seed


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = build_plantvillage_datasets(
        PLANTVILLAGE_TRAIN_DIR,
        PLANTVILLAGE_VAL_DIR,
    )
    train_loader, val_loader = build_dataloaders(
        train_data,
        val_data,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )

    plantdoc_train, plantdoc_test = build_plantdoc_datasets(
        PLANTDOC_ROOT,
        train_data.class_to_idx,
    )

    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"PlantVillage train images: {len(train_data)}")
    print(f"PlantVillage validation images: {len(val_data)}")
    print(f"Number of classes: {len(train_data.classes)}")
    print(f"PlantDoc train images: {len(plantdoc_train)}")
    print(f"PlantDoc test images: {len(plantdoc_test)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("\nPlantVillage train class counts:")
    pv_train_counts = class_counts_from_imagefolder(train_data)
    for idx, name in enumerate(train_data.classes):
        print(f"{idx:2d} {name:50s} {pv_train_counts[idx]:5d}")

    print("\nPlantVillage validation class counts:")
    pv_val_counts = class_counts_from_imagefolder(val_data)
    for idx, name in enumerate(train_data.classes):
        print(f"{idx:2d} {name:50s} {pv_val_counts[idx]:5d}")

    print("\nPlantDoc train class counts:")
    pd_train_counts = class_counts_from_plantdoc(plantdoc_train)
    for idx, name in enumerate(train_data.classes):
        if pd_train_counts[idx]:
            print(f"{idx:2d} {name:50s} {pd_train_counts[idx]:5d}")

    print("\nPlantDoc test class counts:")
    pd_test_counts = class_counts_from_plantdoc(plantdoc_test)
    for idx, name in enumerate(train_data.classes):
        if pd_test_counts[idx]:
            print(f"{idx:2d} {name:50s} {pd_test_counts[idx]:5d}")


if __name__ == "__main__":
    main()
