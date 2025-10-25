"""
Neuronpedia integration for nanochat SAEs.

Provides utilities to upload SAEs to Neuronpedia and retrieve feature descriptions.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
import json

from sae.models import BaseSAE
from sae.config import SAEConfig


class NeuronpediaUploader:
    """Uploader for Neuronpedia integration.

    Note: Actual upload requires manual submission via Neuronpedia's web interface.
    This class prepares the data in the correct format for upload.

    See: https://docs.neuronpedia.org/upload-saes
    """

    def __init__(
        self,
        model_name: str = "nanochat",
        model_version: str = "d20",
    ):
        """Initialize uploader.

        Args:
            model_name: Name of the model (e.g., "nanochat")
            model_version: Version/size of model (e.g., "d20", "d26")
        """
        self.model_name = model_name
        self.model_version = model_version

    def prepare_sae_for_upload(
        self,
        sae: BaseSAE,
        config: SAEConfig,
        output_dir: Path,
        metadata: Optional[Dict] = None,
    ):
        """Prepare SAE for Neuronpedia upload.

        Creates directory with all necessary files for upload:
        - SAE weights
        - Configuration
        - Metadata
        - README with upload instructions

        Args:
            sae: Trained SAE model
            config: SAE configuration
            output_dir: Directory to save upload files
            metadata: Additional metadata (training details, performance metrics, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save SAE weights
        sae_path = output_dir / "sae_weights.pt"
        torch.save({
            "W_enc": sae.W_enc.cpu(),
            "b_enc": sae.b_enc.cpu(),
            "W_dec": sae.W_dec.cpu(),
            "b_dec": sae.b_dec.cpu(),
        }, sae_path)

        # Save configuration
        config_data = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "hook_point": config.hook_point,
            "d_in": config.d_in,
            "d_sae": config.d_sae,
            "activation": config.activation,
            "k": config.k if config.activation == "topk" else None,
            "l1_coefficient": config.l1_coefficient if config.activation == "relu" else None,
            "normalize_decoder": config.normalize_decoder,
        }

        if metadata:
            config_data["metadata"] = metadata

        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Create README with upload instructions
        readme_path = output_dir / "README.md"
        readme_content = self._generate_upload_readme(config)
        with open(readme_path, "w") as f:
            f.write(readme_content)

        print(f"Prepared SAE for upload in {output_dir}")
        print(f"Follow instructions in {readme_path} to upload to Neuronpedia")

    def _generate_upload_readme(self, config: SAEConfig) -> str:
        """Generate README with upload instructions."""
        return f"""# Neuronpedia Upload Instructions

## SAE Details

- **Model**: {self.model_name} ({self.model_version})
- **Hook Point**: {config.hook_point}
- **Input Dimension**: {config.d_in}
- **SAE Dimension**: {config.d_sae}
- **Activation Type**: {config.activation}

## Upload Steps

1. Go to https://docs.neuronpedia.org/upload-saes

2. Fill out the submission form with the following information:
   - Model: {self.model_name}
   - Version: {self.model_version}
   - Hook Point: {config.hook_point}
   - SAE Architecture: {config.activation}
   - Expansion Factor: {config.d_sae / config.d_in}x

3. Upload the following files:
   - `sae_weights.pt`: SAE weights
   - `config.json`: Configuration file

4. Submit the form

5. The Neuronpedia team will process your submission within 72 hours

## Using the API

Once uploaded, you can access features via the Neuronpedia API:

```python
# First, install the neuronpedia package (if available)
# pip install neuronpedia

# Then use it (example):
from neuronpedia import get_feature

feature = get_feature(
    model="{self.model_name}-{self.model_version}",
    layer="{config.hook_point}",
    feature_index=4232
)
print(feature.description)
```

## Documentation

- Neuronpedia Docs: https://docs.neuronpedia.org
- Upload Guide: https://docs.neuronpedia.org/upload-saes
- API Docs: https://docs.neuronpedia.org/api
"""


class NeuronpediaClient:
    """Client for interacting with Neuronpedia API.

    Note: This is a placeholder implementation. The actual Neuronpedia API
    may require authentication and have different endpoints.

    For the real implementation, install the neuronpedia package:
    pip install neuronpedia
    """

    def __init__(self, model_name: str = "nanochat", model_version: str = "d20"):
        """Initialize Neuronpedia client.

        Args:
            model_name: Model name
            model_version: Model version
        """
        self.model_name = model_name
        self.model_version = model_version

        # Try to import neuronpedia package if available
        try:
            # This is hypothetical - actual package may have different API
            import neuronpedia
            self.neuronpedia = neuronpedia
            self.available = True
        except ImportError:
            self.neuronpedia = None
            self.available = False
            print("Warning: neuronpedia package not installed. Install with: pip install neuronpedia")

    def get_feature_description(
        self,
        hook_point: str,
        feature_idx: int,
    ) -> Optional[str]:
        """Get auto-generated description for a feature.

        Args:
            hook_point: Hook point (e.g., "blocks.10.hook_resid_post")
            feature_idx: Feature index

        Returns:
            Feature description if available, None otherwise
        """
        if not self.available:
            return None

        # Placeholder implementation
        # Real implementation would make API call to Neuronpedia
        print(f"Getting description for {self.model_name}-{self.model_version}/{hook_point}/feature_{feature_idx}")
        return None

    def get_feature_metadata(
        self,
        hook_point: str,
        feature_idx: int,
    ) -> Optional[Dict]:
        """Get metadata for a feature from Neuronpedia.

        Args:
            hook_point: Hook point
            feature_idx: Feature index

        Returns:
            Feature metadata if available, None otherwise
        """
        if not self.available:
            return None

        # Placeholder implementation
        return None

    def search_features(
        self,
        query: str,
        hook_point: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search for features by semantic query.

        Args:
            query: Search query (e.g., "features related to negation")
            hook_point: Optional hook point to restrict search
            top_k: Number of results to return

        Returns:
            List of matching features
        """
        if not self.available:
            return []

        # Placeholder implementation
        print(f"Searching for: {query}")
        return []


def create_neuronpedia_metadata(
    sae: BaseSAE,
    config: SAEConfig,
    training_info: Optional[Dict] = None,
    eval_metrics: Optional[Dict] = None,
) -> Dict:
    """Create comprehensive metadata for Neuronpedia upload.

    Args:
        sae: Trained SAE
        config: SAE configuration
        training_info: Training details (epochs, steps, time, etc.)
        eval_metrics: Evaluation metrics (MSE, L0, etc.)

    Returns:
        Metadata dictionary
    """
    metadata = {
        "architecture": {
            "type": config.activation,
            "d_in": config.d_in,
            "d_sae": config.d_sae,
            "expansion_factor": config.d_sae / config.d_in,
            "normalize_decoder": config.normalize_decoder,
        },
        "sparsity_config": {
            "k": config.k if config.activation == "topk" else None,
            "l1_coefficient": config.l1_coefficient if config.activation == "relu" else None,
        },
    }

    if training_info:
        metadata["training"] = training_info

    if eval_metrics:
        metadata["evaluation"] = eval_metrics

    return metadata
