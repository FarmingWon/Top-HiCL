# Top-HiCL
This repository provides the official implementation of the paper:
> Top-HiCL: Topology-Aware Hierarchical Contrastive Learning for Enhanced Job Matching

## Overview
We propose a contrastive representation learning framework that captures path-aware, depth-sensitive, and degree-informed signals within a heterogeneous job–skill bipartite graph, named **Topology-Aware Hierarchical Contrastive Learning for Enhanced Job Matching (Top-HiCL)**.
  
![Top-HiCL](https://github.com/user-attachments/assets/fd8ad21b-c8eb-4e0d-86c8-462ea9a98013)

## Dataset
The required dataset files should be placed in the `dataset/` directory, organized as follows:
<!-- Dataset is private -->
```
dataset/
├── hierarchy_dataset.pickle
├── bipartite_dataset.pickle
└── resume_dataset.pickle
```

## Train

```bash
run_all.bat
```
