import torch
from pathlib import Path
# import clip
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np
import torch.nn.functional as F

# Base Wrapper Class
class VisionTransformerWrapper:
    def __init__(self, model_name, device, smaller_edge_size=224, half_precision=False):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):
        dinov2_source = (
            Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        )
        if not dinov2_source.is_dir():
            raise FileNotFoundError(
                "Local DINOv2 torch-hub source was not found: "
                f"{dinov2_source}"
            )
        model = torch.hub.load(
            str(dinov2_source),
            self.model_name,
            source="local",
        )
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        
        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size
    
    def extract_features(self, image_tensor, feature_list=None, cls=False):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)
            if cls == False:
                if feature_list == None:
                    tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
                    return tokens.cpu().numpy()
                else:
                    tokens = self.model.get_intermediate_layers(image_batch, n=feature_list)
                    tokens = list(tokens)
                    return tokens
            else:
                if feature_list == None:
                    outputs = self.model.get_intermediate_layers(image_batch, return_class_token=True)[0]
                    tokens, class_token = outputs[0], outputs[1]
                    return tokens.cpu().squeeze().numpy(), class_token.squeeze().cpu().numpy()
                else:
                    outputs = self.model.get_intermediate_layers(image_batch, n=feature_list, return_class_token=True)
                    tokens = []
                    class_token = []
                    for output in outputs:
                        tokens.append(output[0])
                        class_token.append(output[1].squeeze().cpu().numpy())
                    tokens = list(tokens)
                    class_token = class_token[-1]
                    return tokens, class_token

                
    def get_embedding_visualization(self, tokens, grid_size, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens
        
    def MLMP(self, features_list, grid_size, scales=[1, 5]):

        H, W = grid_size
        fused_features = []

        for feat in features_list:
            B, N, C = feat.shape
            assert N == H * W, f"Mismatch: N={N}, but H×W={H}×{W}={H*W}"
            feat = feat.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

            multi_scale_feats = [feat]
            for s in scales:
                pooled = F.adaptive_avg_pool2d(feat, (s, s))
                upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
                multi_scale_feats.append(upsampled)
            
            fused = sum(multi_scale_feats)
            fused_features.append(fused)

        final_feature = sum(fused_features) / len(fused_features)  
        return final_feature.view(B, C, -1).permute(0, 2, 1).squeeze().cpu().numpy()
      
def get_model(model_name, device, smaller_edge_size=672):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")


    if model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, device, smaller_edge_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
