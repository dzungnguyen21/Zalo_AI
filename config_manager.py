"""
Configuration Loader and Manager
Supports loading from YAML config files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        # Default configuration
        self.config = self._get_default_config()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'dataset': {
                'train_dir': './train',
                'test_dir': './public_test',
                'annotations_path': './train/annotations/annotations.json'
            },
            'models': {
                'feature_model': 'clip',
                'detector_model': 'yolov8x.pt'
            },
            'detection': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'matching': {
                'similarity_threshold': 0.5,
                'detection_weight': 0.5,
                'similarity_weight': 0.5,
                'use_temporal': True,
                'temporal_window': 5,
                'temporal_weight': 0.3
            },
            'tracking': {
                'enabled': True,
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3,
                'feature_threshold': 0.4,
                'feature_weight': 0.5
            },
            'processing': {
                'frame_skip': 1,
                'max_frames': None,
                'device': 'cuda',
                'batch_size': 1
            },
            'output': {
                'predictions_path': 'predictions.json',
                'evaluation_path': 'evaluation_results.json',
                'save_visualizations': False,
                'visualization_dir': './visualizations'
            }
        }
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML config file
        """
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge with default config
            self._deep_update(self.config, user_config)
            
            print(f"Configuration loaded from: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration")
    
    def _deep_update(self, base: Dict, update: Dict):
        """
        Deep update dictionary
        
        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by key path
        
        Args:
            key_path: Dot-separated key path (e.g., 'models.feature_model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value by key path
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()
    
    def save(self, output_path: str):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to save config file
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to: {output_path}")
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        self._print_dict(self.config)
        print("="*60)
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """Recursively print dictionary"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    def apply_preset(self, preset_name: str):
        """
        Apply a configuration preset
        
        Args:
            preset_name: Preset name ('fast', 'accurate', or 'balanced')
        """
        presets = {
            'fast': {
                'models': {
                    'feature_model': 'clip',
                    'detector_model': 'yolov8n.pt'
                },
                'detection': {
                    'confidence_threshold': 0.2
                },
                'matching': {
                    'use_temporal': False
                },
                'processing': {
                    'frame_skip': 5
                }
            },
            'accurate': {
                'models': {
                    'feature_model': 'both',
                    'detector_model': 'yolov8x.pt'
                },
                'detection': {
                    'confidence_threshold': 0.15
                },
                'matching': {
                    'similarity_threshold': 0.45,
                    'use_temporal': True
                },
                'processing': {
                    'frame_skip': 1
                }
            },
            'balanced': {
                'models': {
                    'feature_model': 'clip',
                    'detector_model': 'yolov8l.pt'
                },
                'detection': {
                    'confidence_threshold': 0.2
                },
                'matching': {
                    'similarity_threshold': 0.5
                },
                'processing': {
                    'frame_skip': 2
                }
            }
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Choose from: {list(presets.keys())}")
        
        self._deep_update(self.config, presets[preset_name])
        print(f"Applied preset: {preset_name}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--preset", type=str, choices=['fast', 'accurate', 'balanced'],
                       help="Apply preset configuration")
    parser.add_argument("--print", action="store_true",
                       help="Print configuration")
    parser.add_argument("--save", type=str,
                       help="Save configuration to file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config if Path(args.config).exists() else None)
    
    # Apply preset if specified
    if args.preset:
        config.apply_preset(args.preset)
    
    # Print configuration
    if args.print or not args.save:
        config.print_config()
    
    # Save configuration
    if args.save:
        config.save(args.save)


if __name__ == "__main__":
    main()
