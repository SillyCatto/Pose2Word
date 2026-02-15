"""
Dataset Loader for Sign Language Recognition

Combines optical flow features from RAFT and MediaPipe landmarks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class SignLanguageDataset(Dataset):
    """Dataset combining RAFT optical flow and MediaPipe landmarks."""
    
    def __init__(
        self,
        landmarks_dir: Path,
        flow_dir: Optional[Path] = None,
        extract_flow: bool = True,
        num_frames: int = 30,
        flow_feature_dim: int = 32,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            landmarks_dir: Directory containing landmark .npy files organized by class
            flow_dir: Directory containing flow .npy files (optional if extract_flow=True)
            extract_flow: Whether to extract flow on-the-fly
            num_frames: Expected number of frames per sample
            flow_feature_dim: Dimension of pooled flow features
            transform: Optional transform to apply
        """
        self.landmarks_dir = Path(landmarks_dir)
        self.flow_dir = Path(flow_dir) if flow_dir else None
        self.extract_flow = extract_flow
        self.num_frames = num_frames
        self.flow_feature_dim = flow_feature_dim
        self.transform = transform
        
        # Load dataset metadata
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_dataset()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
    
    def _load_dataset(self):
        """Load dataset file paths and create class mapping."""
        # Get all class directories
        class_dirs = [d for d in self.landmarks_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all samples
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Find all landmark files
            landmark_files = sorted(class_dir.glob("*.npy"))
            
            for landmark_file in landmark_files:
                # Construct corresponding flow file path if needed
                flow_file = None
                if self.flow_dir:
                    flow_file = self.flow_dir / class_name / landmark_file.name
                    if not flow_file.exists():
                        flow_file = None
                
                self.samples.append({
                    "landmark_path": landmark_file,
                    "flow_path": flow_file,
                    "class_name": class_name,
                    "class_idx": class_idx
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Get a sample.
        
        Returns:
            Tuple of (features_dict, label) where features_dict contains:
                - "landmarks": (num_frames, 258) landmark features
                - "flow": (num_frames-1, flow_feature_dim) flow features
            label: Class index
        """
        sample = self.samples[idx]
        
        # Load landmarks
        landmarks = np.load(sample["landmark_path"])  # Shape: (30, 258)
        
        # Load or compute flow
        if sample["flow_path"] and sample["flow_path"].exists():
            flow = np.load(sample["flow_path"])
        elif self.extract_flow:
            # For now, use zero flow if not available
            # In production, you'd extract flow here
            flow = np.zeros((self.num_frames - 1, self.flow_feature_dim))
        else:
            flow = np.zeros((self.num_frames - 1, self.flow_feature_dim))
        
        # Convert to tensors
        landmarks_tensor = torch.from_numpy(landmarks).float()
        flow_tensor = torch.from_numpy(flow).float()
        
        # Apply transforms if any
        if self.transform:
            landmarks_tensor = self.transform(landmarks_tensor)
        
        features = {
            "landmarks": landmarks_tensor,
            "flow": flow_tensor
        }
        
        label = sample["class_idx"]
        
        return features, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class."""
        distribution = {cls: 0 for cls in self.classes}
        for sample in self.samples:
            distribution[sample["class_name"]] += 1
        return distribution


def create_data_loaders(
    landmarks_dir: Path,
    flow_dir: Optional[Path] = None,
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        landmarks_dir: Directory with landmark files
        flow_dir: Directory with flow files
        batch_size: Batch size for training
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = SignLanguageDataset(
        landmarks_dir=landmarks_dir,
        flow_dir=flow_dir
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Set False for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


def collate_batch(batch: List[Tuple[Dict[str, torch.Tensor], int]]):
    """
    Custom collate function for batching.
    
    Args:
        batch: List of (features_dict, label) tuples
        
    Returns:
        Batched features and labels
    """
    landmarks = torch.stack([item[0]["landmarks"] for item in batch])
    flows = torch.stack([item[0]["flow"] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    features = {
        "landmarks": landmarks,
        "flow": flows
    }
    
    return features, labels


if __name__ == "__main__":
    print("=" * 60)
    print("Sign Language Dataset Loader Demo")
    print("=" * 60)
    
    # Example usage
    landmarks_path = Path("path/to/landmarks")
    
    if landmarks_path.exists():
        dataset = SignLanguageDataset(landmarks_dir=landmarks_path)
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Classes: {dataset.classes}")
        print(f"\nClass distribution:")
        for cls, count in dataset.get_class_distribution().items():
            print(f"  {cls}: {count}")
        
        # Test loading a sample
        if len(dataset) > 0:
            features, label = dataset[0]
            print(f"\nSample 0:")
            print(f"  Landmarks shape: {features['landmarks'].shape}")
            print(f"  Flow shape: {features['flow'].shape}")
            print(f"  Label: {label} ({dataset.classes[label]})")
    else:
        print(f"\nLandmarks directory not found: {landmarks_path}")
        print("Please update the path in the demo code.")
