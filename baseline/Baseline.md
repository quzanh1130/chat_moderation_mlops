## Baseline for explore data - training - tracking model

1. Create environment for notebook

```bash
conda create -n chat python=3.9 -y
conda activate chat
pip install ipykernel
pip install -r requirements.txt
```

2. Open notebook by VS Code or jupyter notebook

3. Follow step on notebook: [Mlflow set up on AWS](../aws_setup/mlflow_on_aws.md)

4. View evidently Dashboard

```bash
cd baseline
evidently ui
```

### NOTE: Check using GPU
- Install nvidia toolkit
- Install cuda
- Install cudnn
- Export .bashrc file

**Example in .bashrc file**
```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```