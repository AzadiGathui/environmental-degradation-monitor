> Static export of results from `train_gpu.ipynb` (U-Net semantic segmentation, Sen-2 LULC dataset). Generated from the actual completed training run; see the notebook itself for the full code.

# U-Net training — Sen-2 LULC (GPU edition)

Trains a U-Net for semantic segmentation on the Mendeley "Sen-2 LULC" dataset
(64×64 RGB chips, 7 land-cover classes, mask labels 1–7 with no background class).

**Differences from `train.ipynb` (CPU baseline):**
- Masks pre-converted from `.tif` → `.npy` (100× faster reads; eliminates GPU starvation)
- `BATCH_SIZE = 256` (4× larger; keeps the GPU fed, minimal VRAM increase)
- `SUBSET = None` (full 149,600 train / 32,079 val split instead of a 1,200-chip sample)
- `epochs = 40` with early stopping (patience 4); on a GTX 1660 Ti Max-Q expect ~1.5–2.5 hrs total

Run the **Mask pre-conversion** cell once before training, then execute the rest top-to-bottom.

## Imports

## Mask pre-conversion (run once before training)

The `.tif` mask files are decoded by PIL inside a `tf.numpy_function`, which breaks out
of TensorFlow's graph and becomes the dominant training bottleneck on a GPU — the GPU
idles for ~60–100 ms/batch while the CPU reads each TIFF, versus only ~5–15 ms for the
actual forward+backward pass.

Pre-converting all masks to `.npy` (numpy binary format) reduces mask loading to
~0.3 ms/batch (~100× faster), keeping the GPU above 90% utilisation.

**Run the cell below once.** It skips files that are already converted, so it is safe to
re-run. Conversion takes ~15–20 minutes and adds ~800 MB to disk across all three splits.

```text
train: all 149600 masks already converted.
val: all 32079 masks already converted.
test: all 32079 masks already converted.
Pre-conversion complete. Safe to re-run - already-converted files are skipped.
```

## Configuration

Dataset paths, image size, and class names. Mask pixel values are 1–7 (no
background/0 class), so labels get remapped to 0–6 during loading to work
with `sparse_categorical_crossentropy` and `MeanIoU`.

**Key parameters:**
- `IMG_SIZE = 64` — height and width of each chip in pixels. Fixed by the Mendeley dataset; changing it requires rebuilding all chips.
- `N_CHANNELS = 3` — RGB bands only. Extend to 4 to add NIR (B08), but both the dataset and the model input layer must be rebuilt.
- `BATCH_SIZE = 256` — chips processed per forward+backward pass. 4× the CPU baseline (64). On the GTX 1660 Ti Max-Q this uses ~530 MB of the 6 GB VRAM budget, leaving ample headroom. Larger batches amortise the per-batch GPU kernel launch overhead and give smoother gradient estimates over the full dataset.
- `CLASS_NAMES` — the 7 land-cover categories the model outputs, in the order corresponding to 0-indexed mask labels.
- `N_CLASSES = 7` — number of output channels in the final softmax layer; must match `len(CLASS_NAMES)`.

## Pairing images with masks

Images are `.png`, masks are `.tif`, but they share the same filename stem.
Build matched, sorted lists of paths for each split so image[i] and mask[i]
always correspond.

```text
train: 149600 pairs
val:   32079 pairs
test:  32079 pairs
```

## Raw pairing check

Before any pipeline processing (resizing, normalising, the `.npy` swap), plot
a few image/mask pairs straight off disk to confirm stems are matched
correctly — this is the cheapest place to catch a broken pairing, before
sinking hours into a training run on misaligned data.

![01-raw-pairing-check](images/01-raw-pairing-check.png)

## Loading and decoding

Images (PNG) decode natively with `tf.image`. Masks load from pre-converted
`.npy` files via `np.load` inside a `tf.numpy_function`. Images scaled to
[0, 1]; mask labels shifted from 1–7 to 0–6.

**Why `.npy` instead of `.tif`:** PIL TIFF reads inside `tf.numpy_function` are the
primary GPU bottleneck in `train.ipynb` — ~60–100 ms/batch while the GPU forward
pass takes only ~5–15 ms. Pre-converting to `.npy` (numpy binary format) makes each
mask load ~100× faster (~0.3 ms), keeping the GPU at >90% utilisation. Run the
**Mask pre-conversion** cell once before training.

**`tf.clip_by_value(mask, 0, N_CLASSES - 1)`:** A small number of chips contain stray
pixel value 0 (nodata / edge artefact). After the `- 1` shift they become -1, which
crashes `tf.gather` inside the loss. Clamping prevents this.

## Augmentation

Light, spectrally-safe augmentation only (flips and 90-degree rotations) —
no color jitter, since hue/brightness shifts would distort the spectral
signal that later needs to transfer to real Sentinel-2 bands.

**Why only geometric augmentation:** Satellite imagery has consistent spectral
signatures (vegetation reflects NIR, water absorbs NIR, etc.). Random brightness or
hue shifts teach the model to ignore those signatures, harming transfer to real S2 scenes.

**`tf.image.rot90(k=...)`:** `k` is drawn uniformly from {0, 1, 2, 3}, giving 0°, 90°,
180°, or 270° rotations. Combined with two independent flip axes, this produces all 8
distinct orientations (the full dihedral group for a square) — effectively 8× data
diversity from each chip without adding any new files.

## Augmentation check

`augment()` composes an optional horizontal flip with a 90-degree rotation
(`k` = 0-3), which together produce all 8 distinct orientations of the
dihedral group for a square (see the Augmentation section above). Rather
than rely on `augment()`'s random draws — which might not land on every
orientation — enumerate all 8 directly and plot each one against its mask,
to confirm the image and mask stay in sync under every transform.

![02-augmentation-check](images/02-augmentation-check.png)

## Building the `tf.data` pipelines

Train, validation, and test sets as `tf.data.Dataset` objects. Augmentation
and shuffling are only applied to the training set.

**Key parameters:**
- `SUBSET = None` — uses the full 149,600 train and 32,079 val chips. Each epoch has 585 batches of 256 (vs. 19 batches in the CPU baseline). Set to `{'train': N, 'val': M}` to cap training during debugging.
- `buffer_size = 2000` — 10× the CPU baseline (200), giving better shuffle randomisation across the large dataset. Uses ~130 MB of RAM at 64×64×3 uint8.
- `num_parallel_calls = tf.data.AUTOTUNE` — TensorFlow chooses how many CPU threads to use for `map` operations. On a GPU, maximising this is critical to avoid data-pipeline starvation.
- `prefetch(tf.data.AUTOTUNE)` — prepares the next batch while the GPU processes the current one, hiding data-loading latency.
- `seed=0` in `subsample` — fixed seed so the same chips are sampled on every run; change the seed to get a different random split.

```text
training on 149600 / 149600 train chips, 32079 / 32079 val chips
```

## Sanity check

Decode a few batches and plot an image/mask pair to confirm the pipeline
produces correctly-shaped, correctly-aligned data before training.

```text
image batch: (256, 64, 64, 3) <dtype: 'float32'>
mask batch:  (256, 64, 64, 1) <dtype: 'int32'> unique: [0 1 2 3 4 5 6]
```

![03-pipeline-sanity-check](images/03-pipeline-sanity-check.png)

## Class weights

A spot check of sampled masks showed heavy class imbalance (agriculture and
built-up dominate pixel counts; water and dense forest are minorities).
Plain cross-entropy biases toward majority classes, so compute inverse-frequency
weights from a sample of training masks to use in the loss.

**Key parameters:**
- `sample_size = 5000` — number of masks read to estimate class frequencies. 5,000 out of 149,600 (~3.3%) gives a stable estimate; increasing it gives marginally better weights at the cost of longer startup.
- `freq + 1e-6` — epsilon prevents division by zero for any class absent from the sample.
- `weights / weights.sum() * n_classes` — rescales weights so they average to 1.0. Without this, the total loss magnitude changes with the number of classes, making the learning rate harder to interpret.
- **Reading the output:** `agriculture: 0.062` means agriculture pixels contribute 0.062× the average weight. `water: 2.823` means water pixels contribute 2.823× — the model is penalised ~45× more per water pixel than per agriculture pixel.

```text
          water: 2.823
   dense forest: 2.480
  sparse forest: 0.534
    barren land: 0.284
       built up: 0.108
    agriculture: 0.062
    fallow land: 0.709
```

## U-Net architecture

A 4-level encoder/decoder (32→64→128→256 filters) with skip connections.
Original draft used a 5-level, 1024-filter U-Net sized for 256×256 inputs;
that's oversized for these 64×64 chips, so it's scaled down here.

**Architecture decisions:**
- `base filters = 32` — feature maps in the first conv block, doubling each level (32→64→128→256). For 64×64 chips, 32 base filters is appropriate — a 5-level 1024-filter network would have more parameters than training pixels.
- **4 encoder levels** — each halves spatial resolution via `MaxPooling2D(2)`. At 64×64 input, 4 levels gives an 8×8 bottleneck. Going deeper would reduce it to 4×4 or 2×2, losing spatial context.
- **Skip connections** (`layers.concatenate`) — pass encoder feature maps directly to the matching decoder level. This preserves fine spatial detail (e.g., class boundaries) lost during downsampling.
- **`Conv2DTranspose(stride=2)`** — learned upsampling. The upsampling kernel weights are trained alongside the rest of the network, generally outperforming fixed `UpSampling2D` for segmentation.
- `activation='softmax'` on output — produces a valid probability distribution across `N_CLASSES` for each pixel. Required by `sparse_categorical_crossentropy`.

```text
Model: "unet"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 64, 64, 3) │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (Conv2D)     │ (None, 64, 64,    │        896 │ input_layer[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (Conv2D)   │ (None, 64, 64,    │      9,248 │ conv2d[0][0]      │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (None, 32, 32,    │          0 │ conv2d_1[0][0]    │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (Conv2D)   │ (None, 32, 32,    │     18,496 │ max_pooling2d[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_3 (Conv2D)   │ (None, 32, 32,    │     36,928 │ conv2d_2[0][0]    │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (None, 16, 16,    │          0 │ conv2d_3[0][0]    │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_4 (Conv2D)   │ (None, 16, 16,    │     73,856 │ max_pooling2d_1[… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_5 (Conv2D)   │ (None, 16, 16,    │    147,584 │ conv2d_4[0][0]    │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (None, 8, 8, 128) │          0 │ conv2d_5[0][0]    │
│ (MaxPooling2D)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_6 (Conv2D)   │ (None, 8, 8, 256) │    295,168 │ max_pooling2d_2[… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_7 (Conv2D)   │ (None, 8, 8, 256) │    590,080 │ conv2d_6[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose    │ (None, 16, 16,    │    131,200 │ conv2d_7[0][0]    │
│ (Conv2DTranspose)   │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (None, 16, 16,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 256)              │            │ conv2d_5[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_8 (Conv2D)   │ (None, 16, 16,    │    295,040 │ concatenate[0][0] │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_9 (Conv2D)   │ (None, 16, 16,    │    147,584 │ conv2d_8[0][0]    │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_1  │ (None, 32, 32,    │     32,832 │ conv2d_9[0][0]    │
│ (Conv2DTranspose)   │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_1       │ (None, 32, 32,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 128)              │            │ conv2d_3[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (Conv2D)  │ (None, 32, 32,    │     73,792 │ concatenate_1[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (Conv2D)  │ (None, 32, 32,    │     36,928 │ conv2d_10[0][0]   │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_2  │ (None, 64, 64,    │      8,224 │ conv2d_11[0][0]   │
│ (Conv2DTranspose)   │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_2       │ (None, 64, 64,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 64)               │            │ conv2d_1[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_12 (Conv2D)  │ (None, 64, 64,    │     18,464 │ concatenate_2[0]… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_13 (Conv2D)  │ (None, 64, 64,    │      9,248 │ conv2d_12[0][0]   │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_14 (Conv2D)  │ (None, 64, 64, 7) │        231 │ conv2d_13[0][0]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 5,777,399 (22.04 MB)
 Trainable params: 1,925,799 (7.35 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 3,851,600 (14.69 MB)
```

## Loss function

Weighted sparse categorical cross-entropy (using the class weights computed
above) combined with a Dice loss term, which helps further with the
imbalance by directly optimising class overlap rather than per-pixel likelihood.

**`weighted_sce`:** Standard cross-entropy where each pixel's loss is multiplied by the
weight of its true class. Rare classes (water, dense forest) have high weights, so the
model is penalised more for getting them wrong.

**`dice_loss`:**
- Computes the Sørensen–Dice coefficient per class per batch: `(2 × intersection + ε) / (sum + ε)`.
- Loss = `1 − mean(Dice across classes)`. A perfect prediction gives Dice = 1 and loss = 0.
- `smooth = 1e-6` — prevents 0/0 when a class is absent from an entire batch.
- Dice loss is not affected by class imbalance in the same way cross-entropy is — it measures overlap per class, so a class with 1% of pixels gets equal weight to one with 50%. This makes it complementary to the weighted SCE.

**`combined_loss = weighted_sce + dice_loss`:** Equal weighting. If a minority class (e.g., water) remains low-IoU after a full run, try `weighted_sce + 2 * dice_loss` to push harder on overlap.

## Compiling the model

Optimise with Adam. Track mean IoU rather than pixel accuracy — accuracy is
misleading on an imbalanced segmentation task (a model predicting only
"agriculture" everywhere would still score ~40% accuracy).

**`Adam(learning_rate=1e-3)`:** Adaptive Moment Estimation. Maintains per-parameter
learning rates based on first and second gradient moments. `1e-3` is Adam's canonical
default and a good starting point for most segmentation tasks. The `ReduceLROnPlateau`
callback will automatically reduce this when training plateaus.

**`MeanIoU(num_classes=7, sparse_y_true=True, sparse_y_pred=False)`:**
- `sparse_y_true=True` — ground-truth masks are integer class indices, not one-hot. Avoids expanding a `(B, 64, 64, 1)` tensor to `(B, 64, 64, 7)`.
- `sparse_y_pred=False` — predictions are softmax probabilities `(B, 64, 64, 7)`; MeanIoU internally takes the argmax.
- Mean IoU averages per-class IoU across all 7 classes equally, giving the same weight to rare and common classes — the right metric when minority-class performance matters.

## Callbacks

Stop early and reduce the learning rate when validation mean IoU plateaus;
checkpoint the best weights seen during training.

**`EarlyStopping`:**
- `monitor='val_mean_iou', mode='max'` — watches validation IoU and stops when it stops improving (higher = better, hence `max`).
- `patience=4` — waits 4 epochs of no improvement before stopping. Tight enough to avoid wasting compute, loose enough to survive a temporary plateau after a learning-rate reduction.
- `restore_best_weights=True` — rewinds model weights to the best epoch after stopping, so the saved model is not the last (possibly slightly worse) checkpoint.

**`ReduceLROnPlateau`:**
- `factor=0.5` — halves the learning rate on plateau detection. Starting at 1e-3: first reduction → 5e-4, second → 2.5e-4, etc.
- `patience=2` — reduces LR after just 2 non-improving epochs (half the EarlyStopping patience), giving the model a chance to recover before stopping entirely.
- `min_lr=1e-6` — floor on the learning rate; below this, further reductions have negligible effect.

**`ModelCheckpoint`:**
- `save_best_only=True` — only overwrites the file when `val_mean_iou` improves, so the file always contains the best weights seen so far.
- Saves in Keras v3 format (`.keras`), which bundles architecture, weights, and optimiser state in a single file.

## Training

Train for up to 40 epochs on the full dataset; early stopping (patience 4 on
validation mean IoU) will halt training earlier if improvement stalls.
On a GTX 1660 Ti Max-Q expect ~3–4 min/epoch → ~1.5–2.5 hrs total
(early stopping typically triggers around epoch 25–35).

**`epochs=40`:** Maximum complete passes through the full 149,600-chip training set.
585 batches of 256 per epoch. With the `.npy` mask fix the GPU should run at
>90% utilisation; if it drops back toward 20%, check that the mask conversion
ran completely for all three splits.

**`history`:** Object returned by `model.fit()`. Stores per-epoch metric values:
- `history.history['loss']` / `'val_loss'` — combined loss per epoch
- `history.history['mean_iou']` / `'val_mean_iou'` — mean IoU per epoch
- `history.history['learning_rate']` — LR at each epoch; shows `ReduceLROnPlateau` reductions

**What to expect vs. the CPU baseline (train.ipynb):** The CPU run hit val IoU ~0.24 at
epoch 15 on only 1,200 chips. A full GPU run should comfortably reach 0.40–0.55 val
mean IoU, with water and dense forest IoU recovering from near-zero to 0.15–0.30 once
the model sees enough rare-class chips in each epoch.

```text
NOTE: this is a reconstruction from the live kernel's `history` object, recovered
after the original console log/output was accidentally cleared. Per-epoch wall-clock
timings (e.g. "72s") were only ever printed to stdout by Keras during training and were
never stored in `history.history`, so exact per-epoch durations could not be recovered
-- everything below is the exact real metric data from the actual run.

Training ran for 38 epochs (of a maximum 40) before EarlyStopping (patience=4) triggered.

epoch     loss  mean_iou  val_loss  val_mean_iou         lr
-----------------------------------------------------------
    1   1.0534    0.2675    1.0380        0.2647   1.00e-03
    2   0.9543    0.3328    0.9951        0.3177   1.00e-03
    3   0.8995    0.3681    0.9488        0.3275   1.00e-03
    4   0.8620    0.3917    0.9404        0.3362   1.00e-03
    5   0.8352    0.3993    0.8930        0.3649   1.00e-03
    6   0.7678    0.4154    0.8503        0.3746   1.00e-03
    7   0.7336    0.4267    0.9021        0.3675   1.00e-03
    8   0.7119    0.4372    0.8193        0.4029   1.00e-03
    9   0.6952    0.4454    0.8078        0.4012   1.00e-03
   10   0.6827    0.4518    0.8199        0.4034   1.00e-03
   11   0.6708    0.4581    0.7800        0.4117   1.00e-03
   12   0.6604    0.4643    0.7494        0.4387   1.00e-03
   13   0.6495    0.4709    0.7568        0.4385   1.00e-03
   14   0.6439    0.4743    0.7839        0.4239   1.00e-03
   15   0.6195    0.4904    0.7332        0.4467   5.00e-04
   16   0.6112    0.4961    0.7036        0.4504   5.00e-04
   17   0.6079    0.4986    0.6998        0.4593   5.00e-04
   18   0.6041    0.5004    0.7206        0.4576   5.00e-04
   19   0.6009    0.5029    0.7564        0.4492   5.00e-04
   20   0.5880    0.5122    0.7230        0.4635   2.50e-04
   21   0.5845    0.5145    0.6870        0.4719   2.50e-04
   22   0.5800    0.5162    0.6897        0.4715   2.50e-04
   23   0.5818    0.5147    0.7022        0.4657   2.50e-04
   24   0.5727    0.5205    0.6529        0.4792   1.25e-04
   25   0.5707    0.5212    0.6892        0.4704   1.25e-04
   26   0.5687    0.5225    0.6539        0.4811   1.25e-04
   27   0.5663    0.5241    0.6693        0.4737   1.25e-04
   28   0.5649    0.5252    0.6577        0.4794   1.25e-04
   29   0.5628    0.5267    0.6519        0.4850   6.25e-05
   30   0.5622    0.5271    0.6452        0.4868   6.25e-05
   31   0.5608    0.5280    0.6622        0.4812   6.25e-05
   32   0.5595    0.5289    0.6732        0.4788   6.25e-05
   33   0.5607    0.5279    0.6549        0.4857   3.13e-05
   34   0.5594    0.5286    0.6325        0.4925   3.13e-05
   35   0.5589    0.5290    0.6431        0.4894   3.13e-05
   36   0.5583    0.5295    0.6401        0.4904   3.13e-05
   37   0.5607    0.5279    0.6365        0.4914   1.56e-05
   38   0.5601    0.5281    0.6457        0.4878   1.56e-05

Best epoch: 34 (val_mean_iou=0.4925) -- EarlyStopping restored weights from this epoch.
```

## Training curves

Plot loss and mean IoU across epochs to check for overfitting or
under-training before evaluating on the held-out test set.

![04-training-curves](images/04-training-curves.png)

## Test set evaluation

Evaluate on the untouched test split for the final reported loss/mean IoU.

```text
test loss: 0.6292
test mean IoU: 0.4931
```

## Per-class IoU

Mean IoU hides which classes the model actually struggles with — recompute
IoU per class over the test set to see if the minority classes (water,
dense forest) are being learned or just ignored.

```text
          water: 0.243
   dense forest: 0.337
  sparse forest: 0.418
    barren land: 0.485
       built up: 0.623
    agriculture: 0.846
    fallow land: 0.499
```

## Confusion matrix — why IoU beats accuracy here

Pixel accuracy divides all correct pixels by all pixels. Since agriculture and
built-up together make up most of the pixels in this dataset, a model can score
high overall accuracy just by getting the common classes right, even while
completely failing on rare classes like water and dense forest.

IoU (`intersection / union`) is computed *per class*, then averaged. It
penalises missed pixels (false negatives) and extra pixels (false positives)
equally, regardless of how common that class is — a class with 1% of pixels
counts exactly as much toward mean IoU as a class with 40% of pixels. That's
why mean IoU can look much lower than accuracy on an imbalanced dataset like
this one, and why it's the more honest metric for judging whether the model
actually learned the rare classes.

The `cm` confusion matrix computed above makes this concrete: the raw counts
are dominated by whichever classes are common, but the row-normalised version
(each row divided by its total — i.e. per-class recall) shows exactly where
each class's pixels are actually ending up.

![05-confusion-matrix](images/05-confusion-matrix.png)

```text
pixel accuracy (this model):                          0.802
mean IoU (this model):                                 0.493
"always predict agriculture" baseline accuracy: 0.490
"always predict agriculture" baseline mean IoU: 0.070
```

## Interpreting the per-class IoU values

Each class's IoU depends on both how well the model learned it and how much
signal it saw during training (governed by the `class_weights` computed
earlier). The cell below ranks classes from weakest to strongest IoU and
checks whether low IoU lines up with high class weight (i.e. rarity):

- **Negative correlation** (rarer classes score lower IoU) means the weighted
  loss + Dice term are pulling in the right direction but haven't fully closed
  the imbalance gap — more epochs, a higher Dice weight, or more aggressive
  class weighting would help.
- **Weak or no correlation** means some classes are weak for reasons other
  than rarity — most likely visual ambiguity with a neighbouring class (check
  the confusion matrix above to see which class it's being confused with).

```text
classes ranked weakest to strongest IoU:
            water: IoU=0.243, class weight=2.82
     dense forest: IoU=0.337, class weight=2.48
    sparse forest: IoU=0.418, class weight=0.53
      barren land: IoU=0.485, class weight=0.28
      fallow land: IoU=0.499, class weight=0.71
         built up: IoU=0.623, class weight=0.11
      agriculture: IoU=0.846, class weight=0.06

weakest class:   water (IoU=0.243)
strongest class: agriculture (IoU=0.846)
correlation between class weight (rarity) and IoU: -0.80
Negative correlation: rarer classes tend to have lower IoU -- the imbalance
is only partly offset by the weighted loss and Dice term.
```

## Visual check

Plot a few test images alongside their ground-truth and predicted masks for
a qualitative sense of where the model succeeds or fails.

![06-ground-truth-vs-prediction](images/06-ground-truth-vs-prediction.png)

## Interpreting the ground-truth vs prediction grid

Beyond the aggregate metrics, it's worth looking at where errors happen
spatially. Look for:

- **Boundary blur** — predictions that get the coarse land-cover right but
  blur or round off sharp edges between classes (a common U-Net artefact at
  64×64 resolution with only 3 downsampling steps).
- **Minority-class dropout** — chips where the ground truth has a patch of a
  rare class (e.g. water) that the prediction misses entirely and instead
  extends a neighbouring majority class over it. This is exactly the failure
  mode a high pixel-accuracy score can hide, but low per-class IoU exposes.
- **Confusion between visually similar classes** — e.g. sparse vs dense
  forest, or barren land vs fallow land, which look similar in colour and
  texture at this resolution.

The cell below quantifies the four specific chips plotted above, so you can
match what you see visually to a number.

```text
per-sample metrics for the four chips plotted above:
  sample 0: pixel accuracy=0.830, mean IoU (classes present)=0.508
  sample 1: pixel accuracy=0.862, mean IoU (classes present)=0.492
  sample 2: pixel accuracy=0.774, mean IoU (classes present)=0.497
  sample 3: pixel accuracy=0.750, mean IoU (classes present)=0.445
```
