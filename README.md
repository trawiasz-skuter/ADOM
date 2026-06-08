# ADOM 26L - Segmentacja Obrazów (U-Net)

Projekt zaliczeniowy z przedmiotu Analiza Danych Obrazowych i Multimedialnych. Repozytorium bazuje na kodzie Pytorch-UNet i zostało dostosowane do eksperymentów ze zdjęciami drogowymi.
Wybrany temat: Temat 11 - Segmentacja obrazów (U-Net, 2015).

## Cel projektu
Implementacja architektury U-Net i ocena eksperymentalna na wybranych obrazach. Podstawowym eksperymentem jest segmentacja binarna w formacie 128x128.

## Instalacja

Sklonuj repozytorium i przygotuj środowisko wirtualne:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Środowisko jest przystosowane do uruchomienia na CPU.

## Uruchomienie

Aby przetestować wycięcie maski na przykładowym zdjęciu drogowym, użyj poniższej komendy. 
Skrypt automatycznie załaduje pre-trenowane wagi:

```bash
python predict.py -i Dataset/image.png -m unet_carvana_scale0.5_epoch2.pth -o Dataset/image_OUT.png
```

## Notatnik demonstracyjny

Przygotowano plik `demo.ipynb`, który ładuje wybrane zdjęcie, skaluje je do 128x128 pikseli, a następnie przepuszcza przez wytrenowany model i wizualizuje wynik predykcji. 
W pliku `demo.html` znajdują się wyniki eksperymentu wraz z ich opracowaniem.
## Trening

Zdjęcia do treningu powinny znajdować się w `data/imgs/`, a odpowiadające im maski w `data/masks/`. 
Aby odpalić trening:

```bash
python train.py --epochs 5 --batch-size 1 --learning-rate 1e-5
```
