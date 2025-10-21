"""
Multimodal model combining vision (ResNet18+CBAM) and text (BERT) for pneumonia classification.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .resnet_cbam import ResNet18CBAM


class MultimodalPneumoniaClassifier(nn.Module):
    """
    Multimodal classifier that fuses visual and textual information.
    
    Uses ResNet18+CBAM for image feature extraction and BERT for text encoding,
    then fuses them for final classification.
    
    Args:
        num_classes: Number of output classes (default: 2)
        pretrained_vision: Whether to use pretrained ResNet18 weights
        bert_model_name: Name of the BERT model to use
        fusion_method: Method to fuse vision and text features ('concat', 'add', 'attention')
        dropout_rate: Dropout rate for the fusion layer
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained_vision: bool = True,
        bert_model_name: str = 'bert-base-uncased',
        fusion_method: str = 'concat',
        dropout_rate: float = 0.3
    ):
        super(MultimodalPneumoniaClassifier, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Vision encoder (ResNet18 + CBAM)
        self.vision_encoder = ResNet18CBAM(num_classes=num_classes, pretrained=pretrained_vision)
        vision_feat_dim = 512  # ResNet18 feature dimension
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        text_feat_dim = self.text_encoder.config.hidden_size  # 768 for bert-base
        
        # Freeze BERT layers (optional - can be fine-tuned)
        for param in self.text_encoder.parameters():
            param.requires_grad = True  # Set to False to freeze
        
        # Feature projection layers
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = 256 * 2
        elif fusion_method == 'add':
            fusion_dim = 256
        elif fusion_method == 'attention':
            fusion_dim = 256
            self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=dropout_rate)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, input_ids=None, attention_mask=None, use_text=True):
        """
        Forward pass through the multimodal model.
        
        Args:
            images: Batch of images (B, C, H, W)
            input_ids: BERT input token ids (B, seq_len)
            attention_mask: BERT attention mask (B, seq_len)
            use_text: Whether to use text features (if False, vision-only)
            
        Returns:
            Classification logits (B, num_classes)
        """
        # Extract vision features
        vision_features = self.vision_encoder.get_features(images)
        vision_features = self.vision_projection(vision_features)
        
        # If text is not provided or use_text is False, use vision-only
        if not use_text or input_ids is None:
            fused_features = vision_features
        else:
            # Extract text features
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Use [CLS] token representation
            text_features = text_outputs.last_hidden_state[:, 0, :]
            text_features = self.text_projection(text_features)
            
            # Fuse features
            if self.fusion_method == 'concat':
                fused_features = torch.cat([vision_features, text_features], dim=1)
            elif self.fusion_method == 'add':
                fused_features = vision_features + text_features
            elif self.fusion_method == 'attention':
                # Use cross-attention
                vision_features_expanded = vision_features.unsqueeze(0)
                text_features_expanded = text_features.unsqueeze(0)
                attended_features, _ = self.attention(
                    vision_features_expanded,
                    text_features_expanded,
                    text_features_expanded
                )
                fused_features = attended_features.squeeze(0)
        
        # Classification
        logits = self.classifier(fused_features)
        return logits


class TextGuidedWeaklySupervisedLearning(nn.Module):
    """
    Text-guided weakly supervised learning framework.
    
    This model uses textual descriptions (e.g., radiology reports) to guide
    the training process, even when only image-level labels are available.
    
    Args:
        vision_model: Pre-initialized vision model
        bert_model_name: Name of the BERT model to use
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        vision_model: nn.Module,
        bert_model_name: str = 'bert-base-uncased',
        num_classes: int = 2
    ):
        super(TextGuidedWeaklySupervisedLearning, self).__init__()
        
        self.vision_model = vision_model
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        
        # Alignment layer to align vision and text features
        self.vision_alignment = nn.Linear(512, 768)
        
        # Text-guided attention
        self.text_guided_attention = nn.MultiheadAttention(768, num_heads=8)
        
        # Final classifier
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, images, input_ids=None, attention_mask=None):
        """
        Forward pass with optional text guidance.
        
        Args:
            images: Batch of images
            input_ids: Optional BERT input token ids
            attention_mask: Optional BERT attention mask
            
        Returns:
            Classification logits
        """
        # Extract vision features
        vision_features = self.vision_model.get_features(images)
        vision_features = self.vision_alignment(vision_features)
        
        if input_ids is not None:
            # Extract text features
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = text_outputs.last_hidden_state[:, 0, :]
            
            # Apply text-guided attention to vision features
            vision_features_expanded = vision_features.unsqueeze(0)
            text_features_expanded = text_features.unsqueeze(0)
            
            attended_features, _ = self.text_guided_attention(
                vision_features_expanded,
                text_features_expanded,
                text_features_expanded
            )
            features = attended_features.squeeze(0)
        else:
            features = vision_features
        
        # Classification
        logits = self.classifier(features)
        return logits


def get_multimodal_model(
    num_classes: int = 2,
    pretrained_vision: bool = True,
    fusion_method: str = 'concat'
) -> MultimodalPneumoniaClassifier:
    """
    Factory function to create a multimodal classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained_vision: Whether to use pretrained vision encoder
        fusion_method: Method to fuse vision and text features
        
    Returns:
        MultimodalPneumoniaClassifier instance
    """
    return MultimodalPneumoniaClassifier(
        num_classes=num_classes,
        pretrained_vision=pretrained_vision,
        fusion_method=fusion_method
    )


def get_tokenizer(bert_model_name: str = 'bert-base-uncased') -> BertTokenizer:
    """
    Get BERT tokenizer for text processing.
    
    Args:
        bert_model_name: Name of the BERT model
        
    Returns:
        BertTokenizer instance
    """
    return BertTokenizer.from_pretrained(bert_model_name)
