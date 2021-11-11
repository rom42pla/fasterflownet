# FasterFlowNet

## Setup

- create a new `conda` environment with `conda create`
- log into your `conda` environment using `conda activate <new_environment>`
- install Python packages using `pip install -r requirements.txt`
- download the datasets into `data` to have a directory tree like the following:
``` 
data
    sintel
        bundler
        flow_code
        test
        training
        README.txt
    .gitkeep
    sintel.py
``` 
## Train

```bash
python train.py [dataset]
```

A list of all the optional parameters can be found in [utils/parsers.py](utils/parsers.py)`

## Test

```bash
python test.py [dataset]
```

A list of all the optional parameters can be found in [utils/parsers.py](utils/parsers.py)`

