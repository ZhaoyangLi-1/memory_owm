# syntax=docker/dockerfile:1.6
# owm_memory (LPWM / C-JEPA memory test on RoboMME): the conda env "memory_owm" reproduced package-for-package.
#
# Build — the context is the project root (~4 MB):
#   cd ~/Projects/owm_memory && docker build -f docker/Dockerfile -t owm_memory:latest .
#
# Run — bulk data (~/data = /common/users/zl1308: robomme_data_h5, owm_memory_data, wandb) is mounted at /data:
#   docker run --rm -it --gpus all --ipc=host --user "$(id -u):$(id -g)" \
#     -v /common/users/zl1308:/data \
#     -v ~/Projects/owm_memory:/workspace/owm_memory \
#     -e WANDB_API_KEY -e OPENAI_API_KEY \
#     owm_memory:latest
#   The second -v (live code instead of the copy baked in) is optional. The entrypoint sources scripts/env.sh, so
#   every command of RUN.md works unchanged, e.g.
#     docker run ... owm_memory:latest python scripts/p56_acceptance.py cjepa --episodes 20
#
# Layout: code in /workspace/{owm_memory,cjepa,lpwm,robomme_benchmark}, data in /data. The host's absolute paths
# (configs/experiment.yaml, scripts/env.sh, paths recorded in run configs) stay valid through symlinks:
#   /common/home/zl1308/Projects -> /workspace      /common/home/zl1308/data, /common/users/zl1308 -> /data
#
# Versions: Python 3.11.16 (conda-forge) + the exact `pip freeze` of memory_owm (184 packages, torch 2.6.0+cu126,
# CUDA runtime from the nvidia-* wheels, so the host only needs an NVIDIA driver >= 560 and nvidia-container-toolkit).
# Third-party repos at the commits the host uses; none of them is modified (RUN.md).

FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# compute + utility for PyTorch, graphics for SAPIEN / ManiSkill's Vulkan renderer (demo generation, GT replay, closed loop)
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# build-essential: Triton compiles its CUDA launcher with gcc at run time, and pip may build a few sdists.
# libvulkan1 / libegl1 / libgl1: SAPIEN rendering; libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1: opencv-python.
# bsdextrautils: `column` (run_floor_ceiling.sh).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
        bsdextrautils \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA Vulkan / EGL loader entries, as in ManiSkill's official Dockerfile (the driver libraries themselves are
# injected by nvidia-container-toolkit when NVIDIA_DRIVER_CAPABILITIES contains "graphics").
COPY <<"EOF" /usr/share/vulkan/icd.d/nvidia_icd.json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}
EOF
COPY <<"EOF" /etc/vulkan/implicit_layer.d/nvidia_layers.json
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
EOF
COPY <<"EOF" /usr/share/glvnd/egl_vendor.d/10_nvidia.json
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

# conda env: same command as on the host (`conda create -n memory_owm -c conda-forge --override-channels python=3.11 pip`),
# with the Python patch version pinned
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -n memory_owm -c conda-forge --override-channels python=3.11.16 pip \
    && /opt/conda/bin/conda clean -afy
ENV OWM_PY=/opt/conda/envs/memory_owm/bin/python \
    PATH=/opt/conda/envs/memory_owm/bin:/opt/conda/bin:$PATH

# `pip freeze` of memory_owm (2026-09-21). Two lines differ from the raw freeze: `packaging` was a conda-build file:// URL
# (-> the same version from PyPI), and robomme is installed from its clone below. setuptools 80.9.0 is the version
# robomme pins (pip freeze does not list it).
COPY <<"EOF" /tmp/requirements.lock.txt
--extra-index-url https://download.pytorch.org/whl/cu126
setuptools==80.9.0
packaging==26.3
absl-py==2.5.0
accelerate==1.5.1
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
anyio==4.15.1
arm_pytorch_utilities==0.5.0
asttokens==3.0.2
attrs==26.1.0
braceexpand==0.1.7
cached-property==2.0.1
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.5.0
cloudpickle==3.1.2
contourpy==1.3.2
cycler==0.12.1
dacite==1.9.2
dm-env==1.6
dm-tree==0.1.10
dm_control==1.0.46
docstring_parser==0.18.0
einops==0.8.1
etils==1.13.0
executing==2.2.1
Farama-Notifications==0.0.6
fast_kinematics==0.2.2
filelock==3.32.6
fonttools==4.63.0
frozenlist==1.8.0
fsspec==2026.7.0
gitdb==4.0.12
GitPython==3.1.62
glfw==2.10.2
googleapis-common-protos==1.75.3
grpcio==1.84.0
gymnasium==0.29.1
h11==0.16.0
h5py==3.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpcore2==2.13.0
httpx==0.28.1
httpx2==2.13.0
huggingface-hub==0.29.3
idna==3.19
imageio==2.37.0
imageio-ffmpeg==0.6.0
importlib_resources==7.1.0
iniconfig==2.3.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
jedi==0.20.0
Jinja2==3.1.6
jiter==0.17.0
kiwisolver==1.5.1
labmaze==1.0.6
lazy-loader==0.5
lightning-utilities==0.15.3
lxml==6.1.3
mani_skill @ git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd
Markdown==3.10.3
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline==0.2.2
mplib==0.1.1
mpmath==1.3.0
mujoco==3.13.0
multidict==6.9.0
munkres==1.1.4
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-ml-py==13.610.43
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
ogbench==1.2.1
omegaconf==2.3.1
openai==3.16.2
opencv-python==4.11.0.86
opentelemetry-api==1.44.0
opentelemetry-exporter-otlp-proto-common==1.44.0
opentelemetry-exporter-otlp-proto-http==1.44.0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
pandas==3.0.6
parso==0.8.7
pexpect==4.9.0
pillow==11.0.0
piqa==1.3.2
platformdirs==4.11.9
pluggy==1.6.0
prompt_toolkit==3.0.53
propcache==0.5.4
protobuf==7.36.1
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.4
pyarrow==25.0.1
pycparser==3.0
pydantic==2.13.5
pydantic_core==2.46.5
pygame==2.6.1
Pygments==2.21.0
pymunk==6.8.0
pynvml==13.0.1
PyOpenGL==3.1.10
pyparsing==3.3.2
pyperclip==1.11.0
PySide6==6.11.2
PySide6_Addons==6.11.2
PySide6_Essentials==6.11.2
pytest==9.1.1
python-dateutil==2.9.0.post0
pytorch-fid==0.3.0
pytorch-kinematics==0.7.5
pytorch-lightning==2.6.6
pytorch-seed==0.2.0
PyWavelets==1.8.0
PyYAML==6.0.2
regex==2026.9.10
requests==2.32.3
safetensors==0.5.3
sapien==3.0.3
scikit-image==0.25.2
scipy==1.15.2
seaborn==0.13.2
shapely==2.1.2
shiboken6==6.11.2
six==1.17.0
smmap==5.0.3
sniffio==1.3.1
stack-data==0.6.3
sympy==1.13.1
tabulate==0.10.0
tensorboard==2.21.0
tensorboard-data-server==0.7.2
tifffile==2025.2.18
timm==1.0.29
tokenizers==0.21.4
toppra==0.6.10
torch==2.6.0+cu126
torchaudio==2.6.0+cu126
torchmetrics==1.9.0
torchvision==0.21.0+cu126
tornado==6.5.8
tqdm==4.67.1
traitlets==5.16.1
transformers==4.50.3
transforms3d==0.4.2
trimesh==5.1.0
triton==3.2.0
truststore==0.10.4
typeguard==4.6.0
typing-inspection==0.4.4
typing_extensions==4.16.0
tyro==1.0.16
unicodedata2==17.0.1
urllib3==2.7.0
uv==0.12.16
wandb==0.30.0
wcwidth==0.8.4
webdataset==1.0.2
Werkzeug==3.1.8
wrapt==2.4.1
xxhash==4.0.1
yarl==1.25.1
zipp==4.1.0
EOF
# --no-deps: install exactly the frozen set, no re-resolution (the host env passes `pip check`, verified below too)
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock.txt

# third-party repos at the host's commits (identical to origin, verified 2026-09-21); cjepa and lpwm are used from
# source via sys.path / PYTHONPATH, robomme is an editable install exactly like on the host
ARG ROBOMME_COMMIT=75ac85d6f1b86884588ea4ec1b5274884be690db
ARG CJEPA_COMMIT=412337d0210bf98cee2ca90c3586ab2ea7ca519e
ARG LPWM_COMMIT=4cf53c403433e64c01652ac2adbec66231a46dea
RUN set -e; \
    clone() { git init -q "$2" && git -C "$2" remote add origin "$1" \
              && git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q FETCH_HEAD; }; \
    clone https://github.com/ZhaoyangLi-1/robomme_benchmark.git /workspace/robomme_benchmark "$ROBOMME_COMMIT"; \
    clone https://github.com/ZhaoyangLi-1/cjepa.git /workspace/cjepa "$CJEPA_COMMIT"; \
    clone https://github.com/ZhaoyangLi-1/lpwm.git /workspace/lpwm "$LPWM_COMMIT"
# The committed robomme pyproject.toml pins torch 2.9.1 / torchvision 0.24.1; the host's clone carries an uncommitted
# edit that pins the versions this env runs (2.6.0 / 0.21.0). Only the package metadata differs (installed --no-deps,
# no code changes); without the same edit `pip check` reports the two pins.
RUN sed -i 's/"torch==2.9.1"/"torch==2.6.0"/; s/"torchvision==0.24.1"/"torchvision==0.21.0"/' \
        /workspace/robomme_benchmark/pyproject.toml \
    && pip install --no-cache-dir --no-deps -e /workspace/robomme_benchmark
RUN pip check \
    && python -c "import cv2, h5py, torch, torchvision, sapien, mani_skill, mplib, robomme, transformers, timm, \
pytorch_lightning, accelerate, webdataset, wandb, openai; print('imports ok, torch', torch.__version__)"

# pretrained weights fetched at run time on the host: DINOv2-small (VideoSAUR encoder, videosaur/modules/encoders.py)
# and torchvision VGG16 (LPWM perceptual loss) — baked in so training needs no internet
ENV HF_HOME=/opt/cache/huggingface \
    TORCH_HOME=/opt/cache/torch \
    HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')" \
    && python -c "import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1')" \
    && chmod -R a+rwX /opt/cache

# host-path compatibility (see the header), a HOME that any --user uid can write to, and wandb under the data volume
# as on the host (WANDB_* variables of the host login shell)
RUN mkdir -p /data /common/home/zl1308 /common/users /home/owm \
    && ln -s /workspace /common/home/zl1308/Projects \
    && ln -s /data /common/home/zl1308/data \
    && ln -s /data /common/users/zl1308 \
    && chmod 1777 /home/owm
ENV HOME=/home/owm \
    WANDB_ROOT=/data/wandb \
    WANDB_DIR=/data/wandb \
    WANDB_CACHE_DIR=/data/wandb/cache \
    WANDB_ARTIFACT_DIR=/data/wandb/artifacts \
    WANDB_DATA_DIR=/data/wandb/data \
    WANDB_CONFIG_DIR=/data/wandb/config

# project code last: it is the layer that changes most often
COPY . /workspace/owm_memory
RUN chmod -R a+rwX /workspace

COPY --chmod=755 <<"EOF" /usr/local/bin/owm-entrypoint
#!/usr/bin/env bash
# same shell environment as `source scripts/env.sh` on the host (interpreter, clean PYTHONPATH, OWM_* paths)
source /workspace/owm_memory/scripts/env.sh
exec "$@"
EOF

WORKDIR /workspace/owm_memory
ENTRYPOINT ["/usr/local/bin/owm-entrypoint"]
CMD ["bash"]
