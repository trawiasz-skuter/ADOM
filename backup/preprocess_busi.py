import os
import glob
import shutil
import numpy as np
from PIL import Image

def main():
    base_dir = 'Dataset/Breast-Ultrasound'
    out_dir = 'Dataset/Breast-Ultrasound-Processed'
    
    out_img = os.path.join(out_dir, 'images')
    out_msk = os.path.join(out_dir, 'masks')
    
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)
    
    classes = ['benign', 'malignant', 'normal']
    count = 0
    
    for c in classes:
        path = os.path.join(base_dir, c)
        if not os.path.exists(path):
            continue
            
        print(f"Processing class: {c}...")
        
        # Znajdź wszystkie pliki, które nie mają 'mask' w nazwie (czyli oryginalne obrazki USG)
        for img_path in glob.glob(os.path.join(path, '*.png')):
            if 'mask' not in img_path:
                basename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(basename)[0]
                
                # Zbuduj nazwę do zapisu – dodajemy nazwę klasy na początek żeby uniknąć konfliktów nazw
                new_basename = f"{c}_{basename}"
                new_img_path = os.path.join(out_img, new_basename)
                
                # Skopiuj oryginalne zdjęcie
                shutil.copy2(img_path, new_img_path)
                
                # Szukaj masek
                search_pattern = os.path.join(path, f"{name_without_ext}*mask*.png")
                mask_files = glob.glob(search_pattern)
                
                # Połącz wszystkie znalezione maski w jedną logiczną sumę
                combined_mask = None
                for mf in mask_files:
                    m = np.array(Image.open(mf).convert('L'))
                    if combined_mask is None:
                        combined_mask = np.zeros_like(m, dtype=bool)
                    combined_mask = np.logical_or(combined_mask, m > 0)
                
                if combined_mask is not None:
                    # Zamień maskę na 0 i 255 i zapisz
                    combined_mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                    new_mask_path = os.path.join(out_msk, new_basename)
                    combined_mask_img.save(new_mask_path)
                
                count += 1
                
    print(f"Pomyślnie przetworzono i skopiowano {count} plików obrazów oraz połączono ich maski.")

if __name__ == '__main__':
    main()
