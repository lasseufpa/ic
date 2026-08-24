# Stable-Baselines3 Installation Guide (Windows, Conda, CPU-only)

**IMPORTANT (Windows users):**\
This guide is written to **avoid Box2D** on Windows, because Box2D
requires native compilation (SWIG + C++ toolchain) and often fails on
student machines.\
All core Stable-Baselines3 examples work **without Box2D**.

This version of the guide **includes full support for
`rl-baselines3-zoo`**, including all additional Python dependencies that
are *not reliably pulled in automatically*.

------------------------------------------------------------------------

## 1. Prerequisites

Before starting, make sure you have:

-   Windows 10 or 11 (64-bit)
-   Anaconda or Miniconda installed
-   Internet connection

Verify Conda is available:

``` powershell
conda --version
```

------------------------------------------------------------------------

## 2. Create a Conda Environment

We recommend Python **3.10** for best compatibility.

``` powershell
conda create -n sb3 python=3.10 -y
conda activate sb3
```

Check Python version:

``` powershell
python --version
```

------------------------------------------------------------------------

## 3. Install PyTorch (CPU-only)

Stable-Baselines3 requires **PyTorch \>= 2.3**.

Install CPU-only PyTorch:

``` powershell
conda install pytorch>=2.3 torchvision torchaudio cpuonly -c pytorch -y
```

Verify PyTorch:

``` powershell
python - << EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

Expected result: - PyTorch version \>= 2.3 - CUDA available: False

------------------------------------------------------------------------

## 4. Install Stable-Baselines3

Install SB3 using pip (recommended inside Conda):

``` powershell
pip install stable-baselines3
```

------------------------------------------------------------------------

## 5. Install Gymnasium Environments (NO Box2D)

### ✅ Recommended environments (safe on Windows)

``` powershell
pip install gymnasium[classic-control]
```

Includes: - CartPole-v1 - MountainCar-v0 - Acrobot-v1

------------------------------------------------------------------------

### ❌ Do NOT install Box2D via pip

**Do not run:**

``` powershell
pip install gymnasium[box2d]
```

or

``` powershell
pip install gymnasium[classic-control,box2d]
```

Reason: - Requires SWIG + C++ compiler - Frequently fails on Windows -
Not required for this course

------------------------------------------------------------------------

## 6. Install rl-baselines3-zoo (REQUIRED dependencies)

Clone the Zoo repository:

``` powershell
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
cd rl-baselines3-zoo
```

Install the base Zoo requirements:

``` powershell
pip install -r requirements.txt
```

⚠️ **Important (Windows / Conda):**\
Even after installing `requirements.txt`, the following packages are
often **missing** and must be installed explicitly.

Install them **exactly in this order**:

``` powershell
pip install huggingface_hub
pip install huggingface_sb3
pip install sb3_contrib
pip install optuna
```

Explanation: - `huggingface_hub`: model download and upload support -
`huggingface_sb3`: SB3--HuggingFace integration - `sb3_contrib`:
experimental SB3 algorithms - `optuna`: hyperparameter optimization
(used by the Zoo)

------------------------------------------------------------------------

## 7. Test rl-baselines3-zoo Training

From the `rl-baselines3-zoo` directory:

``` powershell
python train.py --algo ppo --env CartPole-v1
```

You should see training logs starting without import errors.

------------------------------------------------------------------------

## 8. Final Sanity Check

Run:

``` powershell
python - << EOF
import stable_baselines3
import sb3_contrib
import optuna
from huggingface_hub import HfApi
print("SB3 + Zoo environment OK")
EOF
```

If no errors appear, the installation is successful.

------------------------------------------------------------------------

## 9. Notes for Students

-   Always activate the environment before working:

    ``` powershell
    conda activate sb3
    ```

-   Use **CPU-only** unless explicitly instructed otherwise

-   Do **not** install extra packages unless required

-   Ignore Docker instructions in the Zoo documentation

------------------------------------------------------------------------

## 10. Summary of Commands (SAFE DEFAULT)

``` powershell
conda create -n sb3 python=3.10 -y
conda activate sb3
conda install pytorch>=2.3 torchvision torchaudio cpuonly -c pytorch -y
pip install stable-baselines3
pip install gymnasium[classic-control]

git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
cd rl-baselines3-zoo
pip install -r requirements.txt
pip install huggingface_hub huggingface_sb3 sb3_contrib optuna
```

------------------------------------------------------------------------

End of document.
