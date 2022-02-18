# melanoma_diagnosis

## Install Dependency
1. Environment Setup
`cd hact-net
conda env create -f environment.yml`

2.[DGL](https://www.dgl.ai/pages/start.html)
`conda install -c dglteam dgl-cuda11.3`


## Train Command
`CUDA_VISIBLE_DEVICES=0 python train.py --mg_path /projects/patho4/Kechun/diagnosis/dataset/sure_slices/slice_graphs/cell_graphs --config_fpath ./config/melanoma_mggnn_5_classes_pna.yaml --model_path /projects/patho4/Kechun/diagnosis/checkpoints --in_ram -b 1 --epochs 60 -l 0.0005`