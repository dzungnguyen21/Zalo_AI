"""
Feature Extraction Module for Drone Object Detection
Uses CLIP and DINOv2 for extracting visual features from reference images and video frames
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Union
import clip
from transformers import AutoImageProcessor, AutoModel


class FeatureExtractor:
    """
    Extract visual features using CLIP and DINOv2 models
    """
    
    def __init__(self, model_name: str = "clip", device: str = None):
        """
        Initialize feature extractor
        
        Args:
            model_name: Either 'clip', 'dinov2', or 'both'
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name.lower()
        
        self.clip_model = None
        self.clip_preprocess = None
        self.dinov2_processor = None
        self.dinov2_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load the specified models"""
        if self.model_name in ['clip', 'both']:
            print(f"Loading CLIP model on {self.device}...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
        
        if self.model_name in ['dinov2', 'both']:
            print(f"Loading DINOv2 model on {self.device}...")
            self.dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.dinov2_model.eval()
    
    def extract_clip_features(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        """
        Extract CLIP features from images
        
        Args:
            images: Single PIL Image or list of PIL Images
            
        Returns:
            Tensor of shape (N, feature_dim) with normalized features
        """
        if not isinstance(images, list):
            images = [images]
        
        # Preprocess images
        image_tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensors)
            features = F.normalize(features, dim=-1)
        
        return features
    
    def extract_dinov2_features(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        """
        Extract DINOv2 features from images
        
        Args:
            images: Single PIL Image or list of PIL Images
            
        Returns:
            Tensor of shape (N, feature_dim) with normalized features
        """
        if not isinstance(images, list):
            images = [images]
        
        # Preprocess images
        inputs = self.dinov2_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            # Use CLS token as global feature
            features = outputs.last_hidden_state[:, 0]
            features = F.normalize(features, dim=-1)
        
        return features
    
    def extract_features(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        """
        Extract features using the configured model(s)
        
        Args:
            images: Single PIL Image or list of PIL Images
            
        Returns:
            Tensor of shape (N, feature_dim) with normalized features
        """
        if self.model_name == 'clip':
            return self.extract_clip_features(images)
        elif self.model_name == 'dinov2':
            return self.extract_dinov2_features(images)
        elif self.model_name == 'both':
            clip_features = self.extract_clip_features(images)
            dinov2_features = self.extract_dinov2_features(images)
            # Concatenate and normalize
            features = torch.cat([clip_features, dinov2_features], dim=-1)
            features = F.normalize(features, dim=-1)
            return features
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def extract_reference_features(self, reference_images: List[str]) -> torch.Tensor:
        """
        Extract features from reference object images
        
        Args:
            reference_images: List of paths to reference images
            
        Returns:
            Averaged feature vector of shape (feature_dim,)
        """
        images = [Image.open(img_path).convert('RGB') for img_path in reference_images]
        features = self.extract_features(images)
        
        # Average features from multiple reference images
        avg_features = features.mean(dim=0)
        avg_features = F.normalize(avg_features.unsqueeze(0), dim=-1).squeeze(0)
        
        return avg_features
    
    def extract_patch_features(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """
        Extract features from image patches defined by bounding boxes
        
        Args:
            frame: Image as numpy array (H, W, C) in BGR format
            bboxes: List of bounding boxes as (x1, y1, x2, y2)
            
        Returns:
            Tensor of shape (N, feature_dim) with features for each bbox
        """
        if len(bboxes) == 0:
            return torch.empty(0, self.get_feature_dim()).to(self.device)
        
        patches = []
        for x1, y1, x2, y2 in bboxes:
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract patch and convert to PIL Image
            patch = frame[y1:y2, x1:x2]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patches.append(Image.fromarray(patch_rgb))
        
        if len(patches) == 0:
            return torch.empty(0, self.get_feature_dim()).to(self.device)
        
        return self.extract_features(patches)
    
    def compute_similarity(self, reference_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between reference and candidate features
        
        Args:
            reference_features: Tensor of shape (feature_dim,)
            candidate_features: Tensor of shape (N, feature_dim)
            
        Returns:
            Tensor of shape (N,) with similarity scores
        """
        if candidate_features.shape[0] == 0:
            return torch.empty(0).to(self.device)
        
        # Ensure reference features have batch dimension
        if reference_features.dim() == 1:
            reference_features = reference_features.unsqueeze(0)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            reference_features.expand(candidate_features.shape[0], -1),
            candidate_features,
            dim=1
        )
        
        return similarity
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension for the current model configuration"""
        if self.model_name == 'clip':
            return 512  # CLIP ViT-B/32 feature dimension
        elif self.model_name == 'dinov2':
            return 768  # DINOv2-base feature dimension
        elif self.model_name == 'both':
            return 512 + 768
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")


def load_reference_images(reference_dir: str) -> List[str]:
    """
    Load paths to reference images from a directory
    
    Args:
        reference_dir: Path to directory containing reference images
        
    Returns:
        List of paths to reference images
    """
    import os
    import glob
    
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(reference_dir, ext)))
    
    # Sort for consistency
    image_paths.sort()
    
    return image_paths


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor(model_name="clip")
    
    # Extract reference features
    ref_images = ["path/to/ref1.jpg", "path/to/ref2.jpg", "path/to/ref3.jpg"]
    ref_features = extractor.extract_reference_features(ref_images)
    print(f"Reference features shape: {ref_features.shape}")
    
    # Extract features from a frame
    frame = cv2.imread("path/to/frame.jpg")
    bboxes = [(100, 100, 200, 200), (300, 300, 400, 400)]
    patch_features = extractor.extract_patch_features(frame, bboxes)
    print(f"Patch features shape: {patch_features.shape}")
    
    # Compute similarity
    similarities = extractor.compute_similarity(ref_features, patch_features)
    print(f"Similarities: {similarities}")
