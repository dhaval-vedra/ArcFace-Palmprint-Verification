# 🤖 Pre-trained Model Card

This repository contains the pre-trained weights for the **ArcFace Palmprint Verification** model.

*   **Architecture:** ResNet-18 + ArcFace (512-D Embeddings)
*   **Dataset:** Tongji Contactless Palmprint Dataset (CLAHE Enhanced)
*   **Optimal Threshold:** `0.2773`
*   **File Name:** `best_model.pth` (~53 MB)

## 📥 How to Download the Weights

You can download the model directly using any of the methods below:

### Option 1: Direct Download (Browser)
Click the link below to download directly:
👉 **[Download best_model.pth](https://huggingface.co/dhaval-vedra/ArcFace-Palmprint-Verification/resolve/main/best_model.pth)**

### Option 2: Using `curl` (Terminal)
If you are on Linux/macOS or using Git Bash, run:
```bash
curl -L -o best_model.pth https://huggingface.co/dhaval-vedra/ArcFace-Palmprint-Verification/resolve/main/best_model.pth
```

Option 3: Using Hugging Face CLI

If you have the Hugging Face CLI installed, run:

```bash
hf download hf://dhaval-vedra/ArcFace-Palmprint-Verification/best_model.pth
```

📂 Where to place the file?

After downloading, place the best_model.pth file inside the checkpoints/ directory of the main project:

```text
📦 ArcFace-Palmprint-Verification
 ┣ 📂 checkpoints/
 ┃ ┗ 📜 best_model.pth   <-- Place the downloaded file here
 ┗ 📜 ...
```

🚀 How to use it in Python?

Once the file is in the checkpoints/ folder, you can load it easily:

```python
import torch
from src.model import PalmprintArcFace  # Adjust import based on your code

# Initialize model
model = PalmprintArcFace(num_classes=YOUR_NUM_CLASSES, pretrained=False)

# Load weights
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully!")
```

---

Maintained by Dhaval Vedra
