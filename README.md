# Face landmark detection system

All the relevant implementation details, explanations, and code snippets for this work are consolidated in landmark.ipynb, which serves as the primary reference notebook for this project.

[If landmark.ipynb is showing as invalid document, please access from here](https://drive.google.com/file/d/1qz0ZyvJ5torsEAHpyM1jjsf9l_yCmiZQ/view?usp=sharing)


* [Problem statement](https://docs.google.com/document/d/1DSXm99Fxw7gGm5Tj1VU8NmN31B8XzXf-iHG_tTyiInA/edit?usp=sharing)  
* [Weights pth on drive](https://drive.google.com/file/d/1OlFkfwbNuAGYtx9CdVH1z2k9_YktBbls/view?usp=sharing)
* Samples displayed in notebook [Group](https://drive.google.com/file/d/12VFgZjQI8AhVCKwimzhRgnSsIXTSPlvi/view?usp=sharing) | [Single person](https://drive.google.com/file/d/1TCBsgSvARcn-AwdTjwipW6uHrOPgCOYa/view?usp=sharing)





### Why Face Parsing → Landmark Detection is Better than Direct Landmark Training on Poor Data

#### 1. Data Quality Problem in Landmark Annotation
- Many publicly available landmark datasets are small, inconsistent, or noisy.  
- Training a network directly on such “bad” labels leads to overfitting on noise, poor generalization, and unpredictable predictions.  
- Landmarks are very sensitive; a few pixels of error can significantly affect downstream tasks.

#### 2. Face Parsing Provides Robust Structural Priors
- Face parsing (semantic segmentation of facial regions like eyes, lips, nose, eyebrows) captures the **underlying geometry** of the face.  
- Even if the original landmark annotations are noisy, the segmentation network can learn **consistent region boundaries**, because semantic labels are more robust and easier to annotate or generate synthetically.  
- This gives a more stable representation than sparse landmarks alone.

#### 3. Contour/Centroid Extraction Leverages Structural Information
- Once reliable segmentation masks are available, landmarks can be extracted via **contours** or **centroids**:
  - Centroid of the lip mask → precise mouth center.  
  - Contour of the eye mask → eyelid or pupil landmarks.  
- This approach is **less sensitive to individual annotation errors**, as masks aggregate spatial information over the region.

#### 4. Better Generalization
- Landmarks derived from semantically meaningful regions adapt better to variations in pose, lighting, and occlusion.  
- Direct landmark training on noisy data often fails in edge cases (partially occluded eyes, unusual expressions), because the network lacks structural context.

#### 5. Practical Benefits
- Reduces dependency on high-quality landmark datasets.  
- Leverages large face parsing datasets, including synthetic augmentation.  
- Easier to debug and visualize predictions.  
- Less prone to catastrophic failure: slight mask errors still yield reasonable landmark estimates.

#### 6. Empirical Evidence
- Many state-of-the-art pipelines (3D face reconstruction, AR filters, facial animation) adopt this **two-step approach**: region segmentation → landmark extraction.  
- Especially effective when landmark data is sparse, noisy, or missing for small structures (eyebrows, lip corners, pupils).

---

**Summary:**  
> Instead of asking a network to learn fragile, sparse, and noisy points directly, we teach it the more robust task of segmenting facial regions first. Landmarks then emerge naturally from contours or centroids. This pipeline is **more accurate, robust, interpretable, and generalizable**, particularly in low-quality data scenarios.





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
   (U-Net with ResNet50 encoder)
            │
            ▼
     Predicted Segmentation Mask
            │
            ▼
       Loss Computation
   (Dice + CrossEntropy/ComboLoss)
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
