#!/usr/bin/env python3
"""
Image Classifier for Fruits (CNN with TensorFlow / Keras)
=========================================================
AI Internship Project - Codec Technologies

Classifies fruit images (apple, banana, orange) with a Convolutional Neural Network.

What the script does
--------------------
1. Data pipeline : loads images from a folder (one sub-folder per class), resizes them,
                   normalises pixels to [0, 1] and applies data augmentation
                   (random flip / rotation / zoom / contrast) during training only.
2. Model         : 4 x [Conv2D -> BatchNorm -> MaxPooling] -> Flatten -> Dropout -> Dense -> Softmax.
3. Training      : Adam optimiser + Categorical Crossentropy loss, with early stopping.
4. Evaluation    : training vs validation accuracy/loss curves, classification report
                   and confusion matrix on the validation set.
5. Standalone    : if no dataset folder exists, a small SYNTHETIC fruit dataset is generated
                   automatically, so the script runs end-to-end with no downloads.

Expected folder structure for real data (any number of classes)
----------------------------------------------------------------
    fruit_dataset/
        apple/    img001.jpg, img002.jpg, ...
        banana/   ...
        orange/   ...

Usage
-----
    pip install tensorflow numpy matplotlib pillow scikit-learn
    python fruit_image_classifier.py                           # synthetic demo data
    python fruit_image_classifier.py --data_dir my_fruit_folder --epochs 20
    python fruit_image_classifier.py --predict some_fruit.jpg  # after training

Note: the synthetic images are very easy to separate, so accuracy close to 100% on them
is expected and NOT a measure of real-world performance. Use real photos for your final results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
MOCK_CLASSES = ("apple", "banana", "orange")
OUTPUT_DIR = Path("fruit_outputs")
MODEL_PATH = OUTPUT_DIR / "fruit_classifier.keras"
CLASSES_PATH = OUTPUT_DIR / "class_names.json"


# =============================================================================
# 1. MOCK DATASET GENERATOR (synthetic fruit images drawn with Pillow)
# =============================================================================
def generate_mock_dataset(root: Path, n_per_class: int = 150, size: int = 128, seed: int = SEED) -> None:
    """Create `n_per_class` synthetic images for apple / banana / orange under `root/<class>/`.

    Each image has a random gradient background, noise, and a fruit drawn with random
    colour, size, position and rotation, so the CNN has some variation to learn from.
    """
    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    cx = cy = size / 2

    def background() -> Image.Image:
        top = nrng.integers(140, 256, 3)
        bottom = nrng.integers(140, 256, 3)
        t = np.linspace(0, 1, size)[:, None, None]
        grad = np.broadcast_to(top * (1 - t) + bottom * t, (size, size, 3)).copy()
        grad += nrng.normal(0, 6, grad.shape)
        return Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8)).convert("RGB")

    def draw_apple(d: ImageDraw.ImageDraw) -> None:
        r = size * rng.uniform(0.26, 0.34)
        color = rng.choice([(200, 30, 40), (220, 50, 45), (170, 25, 35), (120, 180, 60)])  # red or green
        d.ellipse([cx - r, cy - r * 0.95, cx + r, cy + r * 1.05], fill=color)
        light = tuple(min(255, c + 60) for c in color)
        d.ellipse([cx - r * 0.55, cy - r * 0.6, cx - r * 0.15, cy - r * 0.25], fill=light)  # highlight
        d.line([cx, cy - r * 0.95, cx + r * 0.1, cy - r * 1.25], fill=(90, 60, 30), width=max(2, int(size * 0.02)))
        d.ellipse([cx + r * 0.1, cy - r * 1.3, cx + r * 0.55, cy - r * 1.05], fill=(60, 150, 60))  # leaf

    def draw_orange(d: ImageDraw.ImageDraw) -> None:
        r = size * rng.uniform(0.27, 0.35)
        color = rng.choice([(255, 140, 0), (250, 120, 10), (255, 165, 30)])
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        dark = tuple(max(0, c - 45) for c in color)
        for _ in range(70):  # peel texture
            a, rr = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random()) * 0.92
            x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=dark)
        d.ellipse([cx - r * 0.12, cy - r * 0.98, cx + r * 0.12, cy - r * 0.8], fill=(70, 140, 50))  # nub

    def draw_banana(d: ImageDraw.ImageDraw) -> None:
        rx, ry = size * rng.uniform(0.34, 0.40), size * rng.uniform(0.26, 0.32)
        width = int(size * rng.uniform(0.12, 0.17))
        color = rng.choice([(250, 225, 50), (245, 210, 40), (255, 235, 80)])
        yc = cy - ry * 0.4
        d.arc([cx - rx, yc - ry, cx + rx, yc + ry], start=15, end=165, fill=color, width=width)
        for ang in (15, 165):  # dark tips
            x = cx + rx * math.cos(math.radians(ang))
            y = yc + ry * math.sin(math.radians(ang))
            d.ellipse([x - width / 3, y - width / 3, x + width / 3, y + width / 3], fill=(90, 60, 30))

    drawers = {"apple": draw_apple, "banana": draw_banana, "orange": draw_orange}

    for cls in MOCK_CLASSES:
        out_dir = root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            drawers[cls](ImageDraw.Draw(layer))
            angle = rng.uniform(0, 360) if cls == "banana" else rng.uniform(-25, 25)
            layer = layer.rotate(angle, resample=Image.BICUBIC)
            shift = int(size * 0.1)
            canvas = background()
            canvas.paste(layer, (rng.randint(-shift, shift), rng.randint(-shift, shift)), layer)
            if rng.random() < 0.3:
                canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
            arr = np.asarray(canvas, dtype=np.float32) * rng.uniform(0.8, 1.15) + nrng.normal(0, 5, (size, size, 3))
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out_dir / f"{cls}_{i:04d}.jpg", quality=90)
    print(f"Generated synthetic dataset: {n_per_class * len(MOCK_CLASSES)} images in '{root}'")


# =============================================================================
# 2. DATA PIPELINE (resize + normalise + augment)
# =============================================================================
def load_datasets(data_dir: Path, img_size: int, batch_size: int, val_split: float = 0.2):
    """Load images from folders -> (train_ds, val_ds, class_names).

    - Resizing happens in image_dataset_from_directory (image_size).
    - Labels are one-hot encoded (label_mode='categorical') for categorical crossentropy.
    - Normalisation and augmentation are layers inside the model (see build_model).
    """
    common = dict(validation_split=val_split, seed=SEED, image_size=(img_size, img_size),
                  batch_size=batch_size, label_mode="categorical")
    train_ds = keras.utils.image_dataset_from_directory(str(data_dir), subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(str(data_dir), subset="validation", **common)
    class_names = train_ds.class_names  # read BEFORE cache/prefetch wrap the dataset

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_augmentation() -> keras.Sequential:
    """Random transformations - only active during training, automatically off at inference."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================
def build_model(num_classes: int, img_size: int) -> keras.Model:
    """CNN: augmentation -> rescaling -> 4 conv blocks -> Flatten -> Dropout -> Dense -> Softmax."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255, name="normalise")(x)          # pixels 0-255 -> 0-1

    for filters in (32, 64, 128, 256):                              # feature extractor
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)                                         # classifier head
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs, outputs, name="fruit_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# 4. TRAINING, EVALUATION AND PLOTS
# =============================================================================
def plot_samples(ds, class_names, path: Path, n: int = 9) -> None:
    images, labels = next(iter(ds.take(1)))
    fig = plt.figure(figsize=(7, 7))
    for i in range(min(n, len(images))):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(class_names[int(np.argmax(labels[i].numpy()))])
        ax.axis("off")
    fig.suptitle("Sample training images")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: keras.callbacks.History, path: Path, show: bool) -> None:
    """Training vs validation accuracy and loss curves."""
    h = history.history
    epochs = range(1, len(h["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, h["accuracy"], label="Training accuracy")
    ax1.plot(epochs, h["val_accuracy"], label="Validation accuracy")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.legend()
    ax2.plot(epochs, h["loss"], label="Training loss")
    ax2.plot(epochs, h["val_loss"], label="Validation loss")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {path}")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_model(model: keras.Model, val_ds, class_names, cm_path: Path) -> None:
    """Classification report + confusion matrix on the validation set."""
    try:
        from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
    except ImportError:
        print("scikit-learn not installed - skipping classification report.")
        return

    y_true, y_pred = [], []
    for images, labels in val_ds:  # collect both in the SAME pass (dataset order can change between passes)
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print("\nClassification report (validation set):")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Fruit classifier - confusion matrix")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")


def predict_image(image_path: str) -> None:
    """Load the saved model and classify a single image file."""
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise SystemExit("No trained model found. Train first: python fruit_image_classifier.py")
    model = keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text())
    h, w = model.input_shape[1:3]
    img = keras.utils.load_img(image_path, target_size=(h, w))
    arr = np.expand_dims(keras.utils.img_to_array(img), axis=0)   # shape (1, h, w, 3), model rescales itself
    probs = model.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    print(f"Prediction: {class_names[order[0]]} ({probs[order[0]]:.1%})")
    for i in order[:3]:
        print(f"  {class_names[i]:<12} {probs[i]:.1%}")


# =============================================================================
# 5. MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="CNN fruit image classifier")
    parser.add_argument("--data_dir", default="fruit_dataset", help="Folder with one sub-folder per fruit class")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=128, help="Images are resized to img_size x img_size")
    parser.add_argument("--predict", help="Path of an image to classify with the saved model")
    parser.add_argument("--no_show", action="store_true", help="Do not open plot windows (still saves PNGs)")
    # parse_known_args() ignores extra arguments injected by Jupyter/Colab,
    # so the script also works when pasted into a notebook cell.
    args, _ = parser.parse_known_args()

    if args.predict:
        predict_image(args.predict)
        return

    keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    # ---- Data ---------------------------------------------------------------
    has_data = data_dir.is_dir() and any(p.is_dir() for p in data_dir.iterdir())
    if not has_data:
        print(f"No dataset found at '{data_dir}' -> generating a synthetic demo dataset.")
        generate_mock_dataset(data_dir, size=args.img_size)
    train_ds, val_ds, class_names = load_datasets(data_dir, args.img_size, args.batch_size)
    print(f"\nClasses: {class_names}")
    CLASSES_PATH.write_text(json.dumps(class_names))
    plot_samples(train_ds, class_names, OUTPUT_DIR / "sample_images.png")

    # ---- Model --------------------------------------------------------------
    model = build_model(num_classes=len(class_names), img_size=args.img_size)
    model.summary()

    # ---- Train --------------------------------------------------------------
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    # ---- Evaluate -----------------------------------------------------------
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f} | validation loss: {val_loss:.4f}")
    plot_history(history, OUTPUT_DIR / "training_curves.png", show=not args.no_show)
    evaluate_model(model, val_ds, class_names, OUTPUT_DIR / "confusion_matrix.png")

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
