import torch
import cv2
import os
import numpy as np

from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
from tqdm import tqdm


class DINOv2Extractor:

    def __init__(self, file_list, img_dir="data/images/", cache_dir="cache", subsample_rate=5, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.file_list = file_list
        self.subsample_rate = subsample_rate

        # create image preprocessor
        print("Loading Image Preprocessor")
        self.preprocessor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

        # create dinov2 model
        print("Loading DINOv2")
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(self.device)

        self.model.eval() 
    
    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, img):
        # assuming the image is in BGR (for compatibility)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)

        inputs = self.preprocessor(images=pil_img, return_tensors="pt").to(self.device)
        output = self.model(**inputs, output_hidden_states=True)

        layer_features = output.hidden_states[9]
        patch_tokens = layer_features[:, 1:, :]
        p = 3.0
        gem_pooled = (patch_tokens.clamp(min=1e-6).pow(p).mean(dim=1)).pow(1./p)
        final_vector = torch.nn.functional.normalize(gem_pooled, p=2, dim=1)
        return final_vector.cpu().numpy().flatten()
    
        # cls_token = output.last_hidden_state[:, 0, :]
        # cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=1) # to simplify cosine similarity

        # return cls_token.cpu().numpy().flatten()

    @torch.no_grad()
    def extract_batch(self, batch_size=32):
        cache_path = os.path.join(self.cache_dir, f"dino_ss{self.subsample_rate}.npy")
        if os.path.exists(cache_path):
            print(f"Loading cached Dino features from {cache_path}")
            features = np.load(cache_path)
            return features

        features = []

        for i in tqdm(range(0, len(self.file_list), batch_size), desc="DINOv2 processing batches"):
            batch_files = self.file_list[i:i+batch_size]
            pil_images = []

            for im_file in batch_files:
                pil_images.append(Image.open(os.path.join(self.img_dir, im_file)))
                
            inputs = self.preprocessor(images=pil_images, return_tensors="pt").to(self.device)
            output = self.model(**inputs, output_hidden_states=True)

            layer_features = output.hidden_states[9]
            patch_tokens = layer_features[:, 1:, :]
            p = 3.0
            gem_pooled = (patch_tokens.clamp(min=1e-6).pow(p).mean(dim=1)).pow(1./p)
            final_vector = torch.nn.functional.normalize(gem_pooled, p=2, dim=1)
            np_features = final_vector.cpu().numpy()

            # cls_token = output.last_hidden_state[:, 0, :]
            # cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=1)

            # np_features = cls_token.cpu().numpy()
            
            features.extend(np_features)

        features = np.array(features)
        np.save(cache_path, features)
        
        return features

    


if __name__ == "__main__":
    extractor = DINOv2Extractor()
    print("Ran successfully")



