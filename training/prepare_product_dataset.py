import os
import random
import shutil
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = (
    r"E:\kagglehub-cache\datasets\paramaggarwal"
    r"\fashion-product-images-small\versions\1"
)

CSV_PATH = os.path.join(DATASET_PATH, "styles.csv")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")

OUTPUT_DIR = r"E:\smart-retail-ai\data\products"

CATEGORIES = [
    "Topwear",
    "Shoes",
    "Bags",
    "Bottomwear",
    "Watches"
]

MAX_PER_CLASS = 3000
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# --------------------------------------------------
# Clear previous prepared dataset
# --------------------------------------------------

def clear_output_directory():
    if os.path.exists(OUTPUT_DIR):
        print("Removing old prepared dataset...")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Create folders
# --------------------------------------------------

def create_directories():

    for split in ["train", "validation"]:

        for category in CATEGORIES:

            path = os.path.join(
                OUTPUT_DIR,
                split,
                category
            )

            os.makedirs(
                path,
                exist_ok=True
            )


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

def load_metadata():

    print("Reading styles.csv...")

    df = pd.read_csv(
        CSV_PATH,
        on_bad_lines="skip"
    )

    df = df[
        df["subCategory"].isin(CATEGORIES)
    ].copy()

    print(
        f"Relevant metadata rows: {len(df)}"
    )

    return df


# --------------------------------------------------
# Locate images
# --------------------------------------------------

def collect_images(df):

    dataset = {}

    print("\nAvailable images:")

    for category in CATEGORIES:

        category_df = df[
            df["subCategory"] == category
        ]

        images = []

        for product_id in category_df["id"]:

            image_path = os.path.join(
                IMAGE_DIR,
                f"{product_id}.jpg"
            )

            if os.path.exists(image_path):
                images.append(image_path)

        random.shuffle(images)

        # Limit large categories
        images = images[:MAX_PER_CLASS]

        dataset[category] = images

        print(
            f"{category:15} {len(images)}"
        )

    return dataset


# --------------------------------------------------
# Split + copy
# --------------------------------------------------

def split_dataset(dataset):

    print("\nCreating train/validation split:\n")

    total_train = 0
    total_validation = 0

    for category, images in dataset.items():

        split_index = int(
            len(images) * TRAIN_RATIO
        )

        train_images = images[:split_index]
        validation_images = images[split_index:]

        train_dir = os.path.join(
            OUTPUT_DIR,
            "train",
            category
        )

        validation_dir = os.path.join(
            OUTPUT_DIR,
            "validation",
            category
        )

        for image_path in train_images:

            shutil.copy2(
                image_path,
                train_dir
            )

        for image_path in validation_images:

            shutil.copy2(
                image_path,
                validation_dir
            )

        total_train += len(train_images)
        total_validation += len(validation_images)

        print(
            f"{category:15} "
            f"Train: {len(train_images):4} | "
            f"Validation: {len(validation_images):4}"
        )

    print("\n-----------------------------")
    print("Total train:", total_train)
    print("Total validation:", total_validation)
    print(
        "Total images:",
        total_train + total_validation
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "\n=== PRODUCT DATASET PREPARATION ===\n"
    )

    clear_output_directory()

    create_directories()

    df = load_metadata()

    dataset = collect_images(df)

    split_dataset(dataset)

    print(
        "\nDataset preparation completed."
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
