# Face landmark detection system

All the relevant implementation details, explanations, and code snippets for this work are consolidated in landmark.ipynb, which serves as the primary reference notebook for this project.

* [Problem statement](https://docs.google.com/document/d/1DSXm99Fxw7gGm5Tj1VU8NmN31B8XzXf-iHG_tTyiInA/edit?usp=sharing)  
* [Weights pth on drive](https://drive.google.com/file/d/1OlFkfwbNuAGYtx9CdVH1z2k9_YktBbls/view?usp=sharing)
* Samples displayed in notebook [Group](https://drive.google.com/file/d/12VFgZjQI8AhVCKwimzhRgnSsIXTSPlvi/view?usp=sharing) | [Single person](https://drive.google.com/file/d/1TCBsgSvARcn-AwdTjwipW6uHrOPgCOYa/view?usp=sharing)

## Training flow
```
Data Preparation
   │
   ├── Load dataset (images + masks)
   │
   ├── Preprocessing (resize, normalize, augment)
   │
   └── DataLoader → Batches
            │
            ▼
       Model Forward Pass
   (U-Net with ResNet50 encoder + skip connections)
            │
            ▼
     Predicted Segmentation Mask
            │
            ▼
       Loss Computation
   (e.g., Dice + CrossEntropy/ComboLoss)
            │
            ▼
      Backpropagation
   (optimizer.step, lr scheduler)
            │
            ▼
        Model Update
            │
            ▼
   ────────────────────────
   Repeat for all epochs
   ────────────────────────
            │
            ▼
       Save Checkpoints
```

## Inference flow
```
Input Image
   │
   ├── Preprocessing (resize, normalize)
   │
   ▼
 Forward Pass through Trained Model
 (U-Net with ResNet50 encoder)
   │
   ▼
 Predicted Segmentation Map
   │
   ├── Post-processing (argmax, thresholding, resizing back)
   │
   ▼
 Final Segmentation Output
   │
   ▼
 Visualization (overlay mask on original image along with FACIAL LANDMARKS)
```
