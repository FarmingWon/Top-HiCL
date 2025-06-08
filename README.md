# Top-HiCL
This repository provides the official implementation of the paper:
> Top-HiCL: Topology-Aware Hierarchical Contrastive Learning for Enhanced Job Matching

## Overview
We propose a contrastive representation learning framework that captures path-aware, depth-sensitive, and degree-informed signals within a heterogeneous job–skill bipartite graph, named **Topology-Aware Hierarchical Contrastive Learning for Enhanced Job Matching (Top-HiCL)**.
  
![Top-HiCL](https://github.com/user-attachments/assets/fd8ad21b-c8eb-4e0d-86c8-462ea9a98013)

## Dataset
We use the **Resume dataset** collected from [LiveCareer](https://www.livecareer.com/), and the **Skill & Job taxonomy dataset** provided by [ESCO](https://esco.ec.europa.eu).

The required dataset files should be placed in the `dataset/` directory, organized as follows:
<!-- Dataset is private -->
```
dataset/
├── hierarchy_dataset.pickle # ESCO Skill Hierarchical Taxonomy 
├── bipartite_dataset.pickle # Bipartite graph of skills and jobs from ESCO
└── resume_dataset.pickle # Resume text data extracted from LiveCareer
```

## Train

```bash
run_all.bat
```
