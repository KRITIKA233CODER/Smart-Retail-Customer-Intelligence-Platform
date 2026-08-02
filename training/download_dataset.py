import os

# IMPORTANT: Set before importing kagglehub
os.environ["KAGGLEHUB_CACHE"] = r"E:\kagglehub-cache"

import kagglehub


print("KaggleHub cache:")
print(os.environ["KAGGLEHUB_CACHE"])


path = kagglehub.dataset_download(
    "paramaggarwal/fashion-product-images-small"
)


print("\nDataset downloaded successfully.")

print("\nDataset path:")
print(path)

print("\nFiles/folders:")
for item in os.listdir(path):
    print(" -", item)