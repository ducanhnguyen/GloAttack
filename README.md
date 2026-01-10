This is the repository for the paper ``Generating adversarial examples based on global frequency domain to attack object detectors''
## Step 1: Environment Setup

Open a terminal and run:

```bash
bash command.sh
````

This script will:

* Download the COCO dataset
* Download YOLOv5
* Install and set up all necessary libraries

---

## Step 2: Configure Attack Methods

Open `main.py` and enable or disable attack methods by setting their flags:

* `True` → enable the attack
* `False` → disable the attack

---

## Step 3: Configure the Proposed Method

Open `myconfig.py` and configure the following parameters:

```python
MODEL_TYPE = ...        # Type of model to evaluate

NUM_IMAGE = 1           # Number of images (only images with mAP > 0 are used)
SAVE_DIR = "out"        # Output directory

ZERO_ATTACK = False
VISUALIZE_FREQUENCY = False   # Frequency-domain visualization (logging)
EXPORT_ADV_FOLDER = False     # Export adversarial images (logging)
EXPORT_ORI_FOLDER = False     # Export original images (logging)
```

### Attack Configurations

Each configuration corresponds to one attack run with a specific perturbation budget (`epsilon`) and number of iterations.

```python
GLOATTACK_CONFIGS = [
    {'epsilon': 0.01, 'max_iters': 500,  'interval': 1},
    {'epsilon': 0.03, 'max_iters': 500,  'interval': 1},
    {'epsilon': 0.05, 'max_iters': 500,  'interval': 1},
    {'epsilon': 0.07, 'max_iters': 500,  'interval': 1},
    {'epsilon': 0.09, 'max_iters': 1000, 'interval': 1},
]
```

---

## Step 4: Run the Program

Execute the main entry point:

```bash
python src/main.py
```

```
```
