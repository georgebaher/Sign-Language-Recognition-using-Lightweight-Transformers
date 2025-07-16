from blendshapes_dataloader import WLASLBlendshapesDataset
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
load_dotenv()

facial_blendshapes_file = os.getenv("WLASL100_FACIAL_BLENDSHAPES_PATH")
metadata_file = os.getenv("WLASL_METADATA_PATH")

blendshape_train_dataset = WLASLBlendshapesDataset(
    facial_blendshapes_parquet_path=facial_blendshapes_file,
    metadata_json_path=metadata_file,
    split="train",
    n_heads=8,  # Example: for a transformer model
    feature_padding_mode="repeat"
)


train_loader = DataLoader(blendshape_train_dataset, batch_size=32, shuffle=True)

# Get a sample batch
features_batch, labels_batch = next(iter(train_loader))

print(f"Batch of features shape: {features_batch.shape}")
print(f"Feature dimension from dataset property: {blendshape_train_dataset.feature_dim}")
