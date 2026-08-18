# DAPS4Massimo - Segmentation GUI MVP

GUI Gradio per confrontare due segmentatori e un consenso DAPS output-level.

Supporta ora cinque sorgenti:

1. `Task09_Spleen` - NIfTI 3D con ground truth in `labelsTr/`.
2. `PASCAL_VOC2012` - immagini VOC con ground truth semantica in `SegmentationClass/`.
3. `DAPS raw_images` - immagini RGB della vecchia pipeline DAPS/COCO.
4. `DAPS test_images` - immagini RGB usate dalla vecchia pipeline di valutazione.
5. `DAPS processed_dataset` - pacchetti `.pt` prodotti da `build_offline_dataset.py`.

## Layout atteso

Se `app.py` si trova in:

```text
src/daps_4_Massimo/app.py
```

allora i default sono:

```text
src/daps_4_Massimo/Task09_Spleen/
src/daps_4_Massimo/VOCdevkit/VOC2012/
src/raw_images/
src/test_images/
src/processed_dataset/
```

Puoi comunque modificare `Dataset root` dalla GUI.

## Installazione

Minimo per aprire la GUI e usare Task09/VOC con baseline demo:

```bash
cd src/daps_4_Massimo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Per usare modelli reali `torchvision:*` e per leggere `processed_dataset/*.pt`:

```bash
pip install -r requirements_torch.txt
python app.py
```

## Modelli disponibili

### Task09_Spleen

- `heuristic_loose`
- `heuristic_strict`
- `pred:<nome>` se metti predizioni in:

```text
src/daps_4_Massimo/predictions/<nome>/spleen_2.nii.gz
```

### VOC / raw_images / test_images / processed_dataset

- `voc_demo_red_green`
- `voc_demo_blue_bright`
- `vocpred:<nome>` se metti predizioni in:

```text
src/daps_4_Massimo/voc_predictions/<nome>/<case_id>.png
src/daps_4_Massimo/voc_predictions/<nome>/<case_id>.npy
```

- `torchvision:deeplabv3_resnet50`
- `torchvision:fcn_resnet50`
- `torchvision:lraspp_mobilenet_v3_large`

La prima esecuzione dei modelli torchvision può scaricare i pesi.

## Metriche

### Task09_Spleen

Mostra:

- IoU 3D
- IoU slice
- Dice 3D
- Agreement M1-M2

### PASCAL VOC2012

Mostra:

- mIoU
- mIoU no-background
- Pixel accuracy
- Agreement M1-M2

I pixel VOC con label `255` vengono ignorati.

### raw_images / test_images

Queste cartelle non hanno ground truth. La GUI mostra quindi metriche di accordo:

- Agreement M1-M2
- Agreement DAPS-M1
- Agreement DAPS-M2

Questi valori non sono IoU contro una verita' a terra: misurano solo quanto le segmentazioni concordano tra loro.

### processed_dataset

La GUI carica i `.pt` della vecchia pipeline DAPS e usa `target_semantic` come reference/pseudo-ground-truth.

Importante: `target_semantic` e' una pseudo-label generata offline, non un'annotazione umana.

## Modalita' DAPS

- `daps_v7_entropy_weighted`: consenso pesato dalla confidenza/entropia.
- `daps_v6_average`: media semplice delle probabilita'.
- `union`: solo binario, utile per Task09.
- `intersection`: solo binario, utile per Task09.

Per immagini RGB multiclasse usa solo:

```text
daps_v7_entropy_weighted
daps_v6_average
```

## Fix note: raw_images/test_images/processed_dataset con torchvision

Questa versione corregge il caso in cui `torchvision:deeplabv3_resnet50` e
`torchvision:fcn_resnet50` fallivano sulle immagini della vecchia pipeline DAPS.
La causa era il preprocessing: la GUI precedente usava `weights.transforms()` di
torchvision, che puo' cambiare la dimensione spaziale dell'immagine prima
ell'inferenza. Poi overlay, DAPS e metriche si aspettavano invece maschere della
stessa dimensione dell'immagine originale.

Ora l'adapter torchvision usa lo stesso schema di `daps_translator_06`:

```text
PIL RGB -> resize 512x512 -> T.ToTensor() -> modello -> logits ridimensionati alla HxW originale
```

In questo modo le probabilita' di DeepLab/FCN, le maschere e l'immagine caricata
hanno sempre dimensioni coerenti.
