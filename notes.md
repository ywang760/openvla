# workign configuration

conda openvla environment installed with the following (output from `pip freeze`):

```
absl-py==2.1.0
accelerate==1.1.1
array_record==0.5.1
astunparse==1.6.3
bitsandbytes==0.44.1
cachetools==5.5.0
certifi @ file:///croot/certifi_1725551672989/work/certifi
charset-normalizer==3.4.0
click==8.1.7
contourpy==1.3.1
cycler==0.12.1
dlimp @ git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46
dm-tree==0.1.8
einops==0.8.0
etils==1.10.0
filelock==3.16.1
flash-attn==2.5.5
flatbuffers==24.3.25
fonttools==4.55.0
fsspec==2024.10.0
gast==0.6.0
google-auth==2.36.0
google-auth-oauthlib==1.2.1
google-pasta==0.2.0
grpcio==1.67.1
h5py==3.12.1
huggingface-hub==0.26.2
idna==3.10
importlib_resources==6.4.5
Jinja2==3.1.4
keras==2.15.0
kiwisolver==1.4.7
libclang==18.1.1
Markdown==3.7
MarkupSafe==3.0.2
matplotlib==3.9.2
ml-dtypes==0.2.0
mpmath==1.3.0
namex==0.0.8
networkx==3.4.2
ninja==1.11.1.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.1.105
oauthlib==3.2.2
OpenEXR==3.3.2
opt_einsum==3.4.0
optree==0.13.1
packaging==24.2
peft==0.11.1
pillow==11.0.0
promise==2.3
protobuf==3.20.3
psutil==6.1.0
pyasn1==0.6.1
pyasn1_modules==0.4.1
pyparsing==3.2.0
python-dateutil==2.9.0.post0
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
requests-oauthlib==2.0.0
rsa==4.9
safetensors==0.4.5
scipy==1.14.1
six==1.16.0
sympy==1.13.1
tensorboard==2.15.2
tensorboard-data-server==0.7.2
tensorflow==2.15.0
tensorflow-addons==0.23.0
tensorflow-datasets==4.9.3
tensorflow-estimator==2.15.0
tensorflow-graphics==2021.12.3
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.16.1
termcolor==2.5.0
timm==0.9.10
tokenizers==0.19.1
toml==0.10.2
torch==2.2.0
torchaudio==2.2.0
torchvision==0.17.0
tqdm==4.67.0
transformers==4.40.1
trimesh==4.5.2
triton==2.2.0
typeguard==2.13.3
typing_extensions==4.12.2
urllib3==2.2.3
Werkzeug==3.1.3
wrapt==1.14.1
zipp==3.21.0
```


init.sh with the following

```
module purge
unset LD_LIBRARY_PATH
unset LD_PRELOAD
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate openvla
```

also install `widowx_envs` and `edgeml` per instructions in readme