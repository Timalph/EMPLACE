# EMPLACE: Self-Supervised Urban Scene Change Detection

This repository contains code and instructions to run EMPLACE introduced in the paper EMPLACE: Self-Supervised Urban Scene Change Detection.

![alt text](./Git_images/Aalsmeerwegbuurt_Oost_20210219_pid_0000_001555_20210420_pid_0003_000108_20220322_pid_0000_000135_20220324_pid_0001_000196_5.jpg)
![alt text](./Git_images/Anjeliersbuurt_20180502_pid_0000_005366_20200424_pid_0000_002234_20210317_pid_0000_002441_20220429_pid_0000_002000_2.jpg)

## AC-1M 
The AC-1M consists of data from the City of Amsterdam. This data is publicly available, but due to their continuously updating people blurring algorithm the latest version of these images have to be downloaded. As such I can not directly distribute the images used for our models. The instructions on how to retrieve the AC-1M are as follows:

1. The metadata for all images in the AC-1M is stored in the AC1M.csv file which is downloadable [here](bit.ly/4qYE61g). This file has four columns: nb (neighbourhood), cluster, filename, url, and metadata_url.
2. If you only want the data you can skip this point: It's recommended to split this file into two csvs: 'AC-1M_train.csv' and 'AC-1M_test.csv' (90/10). Make sure to make the split along the clusters, not along the individual images. A pandas groubpy('cluster') is recommended.
3. Using the url column, the images can be downloaded one by one. These need to be stored in the following directory structure: AC1M/{nb}/{cluster}{filename}. If the data is split like described at 2. this would give AC1M_train/{nb}/{cluster}{filename} and AC1M_test/{nb}/{cluster}{filename}
4. If necessary the antenna can be masked by requesting the heading parameter from the metadata_url. Make sure to mask both the antenna at the heading location as well as 180 degree opposite.

## Train EMPLACE

The bash code used to run all experiments is shown in run_experiments.sh. To run all the experiments:

1. Install the environment from the environment.yml: conda env create -f environment.yml
2. Unzip ATM_square_equirectangular.zip (Buildings) and the ATM_square_equirectangular_boom.zip (Trees)
3. Run all the experiments: bash run_experiments.sh

Additionally, to just train SI-1 EMPLACE on 4 gpus run:

```python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=1 --pos=31 --neg=375 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear```
