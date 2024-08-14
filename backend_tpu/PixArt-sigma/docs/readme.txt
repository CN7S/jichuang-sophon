1. start 'scripts/server.py' and 'scripts/inference_np.py'
```
workdir : /home/linaro/workspace/PixArt-sigma/
python3 scripts/server.py
python3 scripts/inference_np.py
```
2. modify config file at '/sd_data/cache/pixart/model_config.ini'
``` config file format
[config]
cfg_scale = 4.5 # changed by user config
steps = 10 # changed by user config
```

3. modify '/sd_data/cache/pixart/tokens.npy'
``` T5 Tokenizer output, preprocessed by Pi 5
```