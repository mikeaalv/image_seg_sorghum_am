Our deep learning model for image segmentation is built with Detectron2 and has been tested on mutliple dataset. The model has been run mostly on HPC environment and AWS. The following steps will be based on HPC.

# Prerequisite

[Detectron2](https://github.com/facebookresearch/detectron2) and other required packages need to be installed properly with GPU support. You might need to talk with HPC staff about it.

# Simple first run

The segmentation model will be trained with multiple hyperparameters. The run will be based on the scripts [here](../scripts/train)

## Data

The data can be found [here](https://www.dropbox.com/sh/6ohczzdtdhqw7dr/AAC0E9w9gKBs0APQQREOUjx1a?dl=0). It is already separated into train, validation, and test with both images and corresponding segmentaiton files.

## Folder structure

```
    data/
      AM_classify2/
        train/
        validate/
        test/
    submitlist.tab
    am_seg_train.sh
    am_seg_train.py
    parameter_sampler.R (can be find [here](../scripts/utilis))
```

`submitlist.tab` is the hyperparameter setting and an example is [here](../scripts/utilis/submitlist.tab). `am_seg_train.sh` is the resource request script where a p100 GPU is requested and the Detectron2 environment (on Sapelo2) is loaded. `am_seg_train.py` is the major running script. `parameter_sampler.R` is the script to submit jobs in batch.

An inferecnce run will need a folder structure like follows (for example on the test set), where the two scripts can be found [here](../scripts/testset) and the pretrained model can be obtained by contact:

```
    data/
      AM_classify2/
        test/
    am_seg_test.sh
    am_seg_test.py
    pretrained/
        model_best.pth
```

## A first easy run

1. Upload data and script to proper location
2. Log into a proper environment that can run a R script and copy files.
3. Load into proper R environment. For example: `ml load R/4.1.0-foss-2019b`
4. Start R and run parameter_sampler.R
5. There should be mutliple job submitted and queued. You can check the result later.


For the inference example, submit the shell file should work. Or you can run it interactively. 

## Debug the training script.

1. When you connect to the HPC, remember to have `-Y` after `ssh`
2. Log into a proper environment. If you need to observe training for a few epoches, you will need GPU, which might end up having a long wait time.

```
srun --pty -p gpu_p --gres=gpu:P100:1 --mem=20gb --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=12:00:00 --job-name=qlogin /bin/bash -l
```

Otherwise the following works for me:

```
srun --pty  -p inter_p  --mem=102G --nodes=1 --ntasks=1 --cpus-per-task=45 --time=12:00:00 --x11 --job-name=xqlogin --export=TERM,DISPLAY /bin/bash
```

3. Load Detectron2 environment `module load Detectron2/0.3-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0`
4. Run and debug the python script.

## Additional information
1. Follow the Detectron2 colab on this [page](https://github.com/facebookresearch/detectron2) if you haven't.
2. Some resources and explanation can be found [here](https://detectron2.readthedocs.io/en/latest/) and [here](https://github.com/facebookresearch/detectron2/issues). Particularly, the function name can be searched in the documentation. Some explanation of configuration is [here](https://detectron2.readthedocs.io/en/latest/modules/config.html)
