# Consenso DAPS per la segmentazione semantica

Framework Python per confrontare due modelli di segmentazione semantica e combinarne le predizioni attraverso un consenso sulle probabilità di output. L'interfaccia Gradio consente di analizzare singole immagini, visualizzare le segmentazioni ed eseguire valutazioni batch con esportazione dei risultati in Excel.

La segmentazione utilizza le 21 classi PASCAL VOC: 20 categorie di oggetti e il background. Il consenso opera a livello di output e offre due strategie: media delle probabilità (**V6**) e fusione pesata in base all'entropia (**V7**).

## Funzionamento

Ogni immagine viene elaborata dai due modelli selezionati. Ciascun modello restituisce una distribuzione di probabilità sulle classi per ogni pixel; il modulo di consenso combina queste distribuzioni e assegna a ciascun pixel la classe con probabilità maggiore.


Le probabilità hanno forma `[C,H,W]`, dove `C` è il numero di classi e `H,W` sono le dimensioni dell'immagine. Le maschere hanno forma `[H,W]` e contengono gli identificativi delle classi.

Quando il dataset fornisce una reference, le segmentazioni vengono confrontate con essa. In assenza di reference, vengono misurati gli accordi tra i due modelli e il consenso. La fusione non richiede addestramento: utilizza direttamente gli output dei segmentatori.
I modelli torchvision elaborano immagini ridimensionate a 512×512 e normalizzate con i parametri dei pesi pretrained; i logits vengono riportati alla risoluzione originale prima della softmax.

Le metriche calcolate sono mIoU, mIoU senza background e pixel accuracy. Per VOC la reference è una ground truth annotata; per processed_dataset può essere una pseudo-label. Su raw_images e test_images si misura soltanto l'accordo tra segmentazioni.

## Installazione

Le dipendenze sono suddivise in due file:

- `requirements.txt`: Gradio, NumPy, Pillow e openpyxl, per interfaccia, elaborazione delle immagini ed esportazione Excel.
- `requirements_torch.txt`: PyTorch e torchvision, per i segmentatori pretrained e il caricamento dei pacchetti `.pt`.

Aprire un terminale nella cartella del progetto.

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements_torch.txt
.\.venv\Scripts\python.exe app.py
```

### Linux e macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements_torch.txt
.venv/bin/python app.py
```

Aprire nel browser l'indirizzo locale mostrato nel terminale. Per utilizzare soltanto i modelli demo o le predizioni precomputate su immagini e dataset VOC è sufficiente installare `requirements.txt`.

## Organizzazione del progetto

```text
Framework/
├── app.py
├── requirements.txt
├── requirements_torch.txt
├── daps_core/
│   ├── models.py
│   ├── consensus.py
│   ├── metrics.py
│   ├── voc.py
│   └── natural.py
├── batch_evaluator/
│   ├── runner.py
│   ├── excel_export.py
│   └── outputs/
├── datasets/
│   ├── VOCdevkit/VOC2012/
│   ├── raw_images/
│   ├── test_images/
│   └── processed_dataset/
└── voc_predictions/
```

| Componente | Responsabilità |
|---|---|
| `app.py` | Definisce la GUI e collega selezione dei dati, inferenza e visualizzazione |
| `daps_core/models.py` | Carica i modelli e converte le loro uscite in probabilità e maschere |
| `daps_core/consensus.py` | Combina le probabilità con le strategie V6 e V7 |
| `daps_core/metrics.py` | Calcola le metriche di segmentazione |
| `daps_core/voc.py` | Gestisce casi VOC, classi, palette, predizioni esterne e overlay |
| `daps_core/natural.py` | Carica immagini RGB e pacchetti `.pt` |
| `batch_evaluator/runner.py` | Esegue la valutazione dei casi e gestisce avanzamento ed errori |
| `batch_evaluator/excel_export.py` | Scrive risultati e configurazione del batch in Excel |

I percorsi predefiniti dei dataset sono relativi alla cartella di `app.py`. Il campo **Dataset root** permette di selezionare una posizione diversa. La cartella `outputs` viene creata durante l'esportazione. 

## Utilizzo dell'interfaccia

### Valutazione singola

1. Aprire **Single evaluation**.
2. Selezionare dataset, **Dataset root** e, per VOC, split.
3. Premere **Refresh cases/models** per aggiornare immagini e modelli disponibili.
4. Scegliere l'immagine, **Model 1**, **Model 2** e **DAPS mode**.
5. Premere **Run segmentation**.

L'interfaccia mostra l'immagine originale e gli overlay di reference, Model 1, Model 2 e DAPS. La tabella riporta le metriche; il pannello **Details** descrive sorgenti, classi presenti e percentuale di pixel modificati dal consenso rispetto a ciascun modello.

Con reference sono mostrati i tre confronti rispetto alla reference e l'agreement M1–M2. Senza reference, il pannello corrispondente resta neutro e la tabella mostra M1–M2, DAPS–M1 e DAPS–M2.

I modelli inizialmente selezionati sono quelli demo. Per utilizzare i segmentatori pretrained, selezionarli esplicitamente nei menu.

### Valutazione batch

1. Aprire **Batch evaluation**.
2. Impostare dataset, percorso, split, due modelli e strategia di consenso.
3. Premere **Run batch evaluation**.
4. Scaricare il file dal campo **Excel Report**.

La scheda batch ha una configurazione indipendente da quella singola. Elabora sequenzialmente tutti i casi individuati nella sorgente selezionata e aggiorna l'avanzamento. Un errore su un'immagine viene registrato senza interrompere l'elaborazione delle successive.

## Report Excel

I report sono salvati in:

```text
batch_evaluator/outputs/daps_batch_YYYYMMDD_HHMMSS.xlsx
```

Ogni file contiene due fogli:

| Foglio | Contenuto |
|---|---|
| **Results** | ID del caso, stato, errore, presenza della reference e metriche per ogni confronto |
| **Run info** | Dataset, percorso, split, modelli, consenso, conteggi dei casi e data di creazione |

`Results` contiene 22 colonne: quattro descrittive e tre metriche per ciascuno dei sei confronti M1/reference, M2/reference, DAPS/reference, M1–M2, DAPS–M1 e DAPS–M2. Le metriche rispetto alla reference restano vuote quando questa non è disponibile.

Lo stato `completed` indica che il caso è stato elaborato; `Has reference` specifica se è stato possibile confrontarlo con una reference. Per i casi `error`, la colonna `Error` riporta il problema incontrato. Il completamento del batch indica che il report è stato prodotto, anche se alcune righe contengono errori.

I valori esportati permettono analisi successive per immagine, coppia di modelli e strategia di consenso. Medie e deviazioni standard vengono calcolate separatamente a partire dalle righe del report.

