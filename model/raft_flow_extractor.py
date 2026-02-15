"""
RAFT Optical Flow Extractor

Extracts optical flow from consecutive keyframes using pre-trained RAFT model.
Optimized for M4 MacBook with MPS acceleration.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.transforms import functional as F


class RAFTFlowExtractor:
    """Extract optical flow between consecutive frames using RAFT."""
    
    def __init__(
        self, 
        model_size: str = "small",
        device: Optional[str] = None,
        batch_size: int = 4
    ):
        """
        Initialize RAFT flow extractor.
        
        Args:
            model_size: "small" or "large" (small is faster, large is more accurate)
            device: Device to run on ("mps", "cuda", "cpu"). Auto-detected if None.
            batch_size: Number of frame pairs to process at once
        """
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading RAFT-{model_size} model on {self.device}...")
        
        # Load pre-trained RAFT model
        if model_size == "large":
            self.model = raft_large(pretrained=True)
        else:
            self.model = raft_small(pretrained=True)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ RAFT model loaded successfully")
    
    def preprocess_frames(
        self, 
        frames: List[np.ndarray], 
        target_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Preprocess frames for RAFT model.
        
        Args:
            frames: List of frames as numpy arrays (H, W, C)
            target_size: Target size (height, width) for resizing
            
        Returns:
            Tensor of shape (N, 3, H, W) normalized to [0, 1]
        """
        processed = []
        
        for frame in frames:
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, target_size[::-1])  # cv2 uses (W, H)
            
            # Convert to tensor and normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            processed.append(frame_tensor)
        
        return torch.stack(processed)
    
    def extract_flow_from_frames(
        self, 
        frames: List[np.ndarray],
        return_magnitude: bool = True
    ) -> np.ndarray:
        """
        Extract optical flow from a sequence of frames.
        
        Args:
            frames: List of frames (H, W, C)
            return_magnitude: If True, returns flow magnitude and angle. 
                            If False, returns raw flow (u, v)
        
        Returns:
            Flow features of shape (N-1, H, W, 2) where N is number of frames
            If return_magnitude=True: (N-1, H, W, 2) with magnitude and angle
            If return_magnitude=False: (N-1, H, W, 2) with u and v components
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to compute optical flow")
        
        # Preprocess frames
        frame_tensors = self.preprocess_frames(frames).to(self.device)
        
        flows = []
        
        # Process consecutive frame pairs
        with torch.no_grad():
            for i in range(len(frame_tensors) - 1):
                frame1 = frame_tensors[i:i+1]
                frame2 = frame_tensors[i+1:i+2]
                
                # RAFT expects input in [0, 255] range
                frame1_scaled = frame1 * 255.0
                frame2_scaled = frame2 * 255.0
                
                # Compute optical flow
                flow_predictions = self.model(frame1_scaled, frame2_scaled)
                
                # Get the final flow prediction (last element in the list)
                flow = flow_predictions[-1][0]  # Shape: (2, H, W)
                
                # Convert to numpy
                flow_np = flow.cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)
                
                if return_magnitude:
                    # Convert to magnitude and angle
                    magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
                    angle = np.arctan2(flow_np[..., 1], flow_np[..., 0])
                    flow_np = np.stack([magnitude, angle], axis=-1)
                
                flows.append(flow_np)
        
        return np.array(flows)
    
    def extract_flow_from_directory(
        self,
        frames_dir: Path,
        output_dir: Optional[Path] = None,
        file_pattern: str = "*.jpg"
    ) -> np.ndarray:
        """
        Extract optical flow from keyframes in a directory.
        
        Args:
            frames_dir: Directory containing keyframe images
            output_dir: Optional directory to save flow visualizations
            file_pattern: Pattern to match frame files
            
        Returns:
            Flow features array
        """
        # Load frames
        frame_paths = sorted(Path(frames_dir).glob(file_pattern))
        frames = [cv2.imread(str(p)) for p in frame_paths]
        
        if len(frames) < 2:
            raise ValueError(f"Found only {len(frames)} frames. Need at least 2.")
        
        print(f"Extracting flow from {len(frames)} frames...")
        
        # Extract flow
        flows = self.extract_flow_from_frames(frames)
        
        # Optionally save visualizations
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_flow_visualizations(flows, output_dir)
        
        return flows
    
    def _save_flow_visualizations(self, flows: np.ndarray, output_dir: Path):
        """Save optical flow as color-coded visualizations."""
        for i, flow in enumerate(flows):
            # Assuming flow is (H, W, 2) with magnitude and angle
            magnitude = flow[..., 0]
            angle = flow[..., 1]
            
            # Create HSV image
            hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
            hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 179  # Hue: angle
            hsv[..., 1] = 255  # Saturation: full
            hsv[..., 2] = np.clip(magnitude * 10, 0, 255).astype(np.uint8)  # Value: magnitude
            
            # Convert to BGR for saving
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(str(output_dir / f"flow_{i:03d}.jpg"), bgr)
    
    def extract_global_flow_features(
        self, 
        flows: np.ndarray,
        pool_size: Tuple[int, int] = (4, 4)
    ) -> np.ndarray:
        """
        Extract global flow features by pooling.
        
        Args:
            flows: Flow array of shape (N, H, W, 2)
            pool_size: Spatial pooling grid size
            
        Returns:
            Pooled flow features of shape (N, pool_size[0] * pool_size[1] * 2)
        """
        N, H, W, C = flows.shape
        ph, pw = pool_size
        
        # Calculate region sizes
        h_step = H // ph
        w_step = W // pw
        
        pooled_features = []
        
        for flow in flows:
            features = []
            for i in range(ph):
                for j in range(pw):
                    # Extract region
                    region = flow[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step, :]
                    
                    # Average pool
                    avg = np.mean(region, axis=(0, 1))
                    features.extend(avg)
            
            pooled_features.append(features)
        
        return np.array(pooled_features)


def demo_raft_extraction():
    """Demo function showing how to use RAFTFlowExtractor."""
    print("=" * 60)
    print("RAFT Optical Flow Extractor Demo")
    print("=" * 60)
    
    # Initialize extractor
    extractor = RAFTFlowExtractor(model_size="small")
    
    # Create dummy frames for demo
    print("\nCreating dummy frames...")
    frames = []
    for i in range(5):
        # Create a moving pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + i * 50, 240), 50, (255, 0, 0), -1)
        frames.append(frame)
    
    # Extract flow
    print("\nExtracting optical flow...")
    flows = extractor.extract_flow_from_frames(frames)
    print(f"✓ Extracted flow shape: {flows.shape}")
    
    # Extract global features
    print("\nExtracting global flow features...")
    global_features = extractor.extract_global_flow_features(flows, pool_size=(4, 4))
    print(f"✓ Global features shape: {global_features.shape}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_raft_extraction()
