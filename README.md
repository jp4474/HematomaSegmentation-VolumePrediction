# HematomaSegmentation-VolumePrediction

# Folder Structure and Description
```bash
.
├── full_raw
│   ├── training
│   │   ├── mri
│   │   ├── mask
│   │   └── slice
│   ├── validation
│   │   ├── mri
│   │   ├── mask
│   │   └── slice
│   └── test
│       └── slice
└── full_preprocessed
    ├── training
    │   ├── mri
    │   ├── mask
    │   └── slice
    ├── validation
    │   ├── mri
    │   ├── mask
    │   └── slice
    └── test
        ├── mri
        ├── mask
        └── slice
```
- mri: This folder contains MRI (Magnetic Resonance Imaging) scans. Each file in this folder represents a brain MRI scan of an individual.

- mask: This folder contains mask data. In the context of medical imaging, a mask is often a binary image that indicates the regions of interest in the corresponding MRI scan. Each file in this folder represents a mask that corresponds to an MRI scan with the same name without the subscript 'seg'.

- slice: This folder contains .npy files, which are NumPy array files. Each file in this folder is a stacked slice of an MRI scan and its corresponding mask. The MRI scan and mask are sliced into 155 slices along the transverse axis (the z-axis), and these slices are stacked together to form a 3D array of size [2, 240, 240]. This array is then saved as a .npy file. Each .npy file in this folder represents the sliced, stacked data of an MRI scan and its corresponding mask.
