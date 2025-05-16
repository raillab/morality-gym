# Benchmarks
Scripts to benchmark the morality-gym using omnisafe

## Setup

Installation can be done though either conda or docker(TBA) using the instrucitons below

#### Options:

1. **Conda:**

   ```setup_conda_env.sh```

2. **Docker:(TBA)**


# For rendering with omnisafe

```# 
sudo apt update
# Reinstall DRI and GLX support
sudo apt install --reinstall libgl1-mesa-dri libgl1-mesa-glx mesa-utils
# Install Vulkan drivers
sudo apt install mesa-vulkan-drivers mesa-va-drivers mesa-vdpau-drivers
# Check if file exists
ls /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so
# Optionally fix symlinks
sudo ln -sf /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri
# Ensure conda is using the newest version
conda install -c conda-forge libstdcxx-ng

```

## Usage
```# 
Run python trainer.py to launch training runs with assigned configs.

```