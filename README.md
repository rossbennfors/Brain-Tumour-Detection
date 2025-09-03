# Brain Tumour Detection using Deep Learning

A deep learning project for classifying brain MRI scans as **tumorous** or **non-tumorous** using transfer learning and fine-tuning on a **VGG19 convolutional neural network**. The project explores baseline feature extraction, progressive fine-tuning, and overfitting trade-offs, achieving up to **89.3% test accuracy**.

---

## 🔍 Overview

-   **Model architecture:** VGG19 (transfer learning + custom dense layers).
-   **Approach:** Started with frozen VGG19 as a feature extractor, then progressively unfroze layers for fine-tuning.
-   **Dataset:** Brain MRI images, binary classification (tumour vs. non-tumour).
-   **Key techniques:** Data augmentation, class balancing, progressive unfreezing, early stopping, checkpointing.
-   **Result:** Final fine-tuned model reached **89.3% accuracy** on test data.

---

## 📊 Results

| Model Stage             | Validation Accuracy | Test Accuracy |
| ----------------------- | ------------------- | ------------- |
| Baseline (frozen VGG19) | 76.7%               | 75.0%         |
| Fine-tune Stage 1       | 81.1%               | 83.0%         |
| Fine-tune Stage 2       | 82.0%               | 83.5%         |
| Fine-tune Stage 3       | 85.4%               | 87.3%         |
| Final Fine-tuned Model  | 85.4%               | **89.3%**     |

---

## 📖 Reference

Full project write-up: [Deep Learning for Brain Tumour MRI Report](report.pdf)
