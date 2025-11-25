import torch
from PIL import Image
from roadmap.getter import Getter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import os
os.path
def test_transforms():
    # Charge la config
    transform_cfg = OmegaConf.load('config/transform/textured_dwt.yaml')
    print(transform_cfg)
    # Crée les transformations via le getter
    getter = Getter()
    train_transform = getter.get_transform(transform_cfg.train)
    test_transform = getter.get_transform(transform_cfg.test)

    # Charge une image test
    img_path = "../../data/car.jpg"#"../../data/stanforddogs/Images/n02086646-Blenheim_spaniel/n02086646_45.jpg"  # Remplace avec ton chemin
    img = Image.open(img_path).convert('RGB')
    
    # Applique les transformations
    img_train = train_transform(img)
    img_test = test_transform(img)

    # Affiche les résultats
    plt.figure(figsize=(12, 4))
    
    plt.subplot(251)
    plt.title('Original')
    plt.imshow(img)
    
    def normalize_for_display(tensor):
        # Move to CPU and convert to numpy
        img = tensor.cpu().permute(1, 2, 0).numpy()
        # Normalize approximation coefficients to [0, 1]
        if tensor.min() >= 0:  # For approximation coefficients
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:  # For detail coefficients, normalize symmetric around zero
            max_abs = max(abs(img.min()), abs(img.max()))
            img = (img + max_abs) / (2 * max_abs + 1e-8)
        return img

    plt.subplot(252)
    plt.title('Train Transform approx')
    plt.imshow(normalize_for_display(img_train[:, 0]))

    plt.subplot(253)
    plt.title('Train Transform details lh')
    plt.imshow(normalize_for_display(img_train[:, 1]))

    plt.subplot(254)
    plt.title('Train Transform details hl')
    plt.imshow(normalize_for_display(img_train[:, 2]))

    plt.subplot(255)
    plt.title('Train Transform details hh')
    plt.imshow(normalize_for_display(img_train[:, 3]))

    plt.subplot(257)
    plt.title('Test Transform approx')
    plt.imshow(normalize_for_display(img_test[:, 0]))

    plt.subplot(258)
    plt.title('Test Transform details lh')
    plt.imshow(normalize_for_display(img_test[:, 1]))

    plt.subplot(259)
    plt.title('Test Transform details hl')
    plt.imshow(normalize_for_display(img_test[:, 2]))

    plt.subplot(2,5,10)
    plt.title('Test Transform details hh')  
    plt.imshow(normalize_for_display(img_test[:, 3])) 
    
    plt.show()

    # Affiche les shapes
    print(f"Train transform output shape: {img_train.shape}")
    print(f"Test transform output shape: {img_test.shape}")

if __name__ == "__main__":
    test_transforms()