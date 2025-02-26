# EMPLACE: Self-Supervised Urban Scene Change Detection

This repository contains code and instructions to run EMPLACE introduced in the paper EMPLACE: Self-Supervised Urban Scene Change Detection. For instructions on retrieving the AC-1M dataset please mail the authors at t.o.l.alpherts@uva.nl.

![alt text](./Git_images/Aalsmeerwegbuurt_Oost_20210219_pid_0000_001555_20210420_pid_0003_000108_20220322_pid_0000_000135_20220324_pid_0001_000196_5.jpg)
![alt text](./Git_images/Anjeliersbuurt_20180502_pid_0000_005366_20200424_pid_0000_002234_20210317_pid_0000_002441_20220429_pid_0000_002000_2.jpg)

The bash code used to run all experiments is shown in run_experiments.sh. To run all the experiments:

1. Install the environment from the environment.yml: conda env create -f environment.yml
2. Unzip ATM_square_equirectangular.zip (Buildings) and the ATM_square_equirectangular_boom.zip (Trees)
3. Run all the experiments: bash run_experiments.sh


To train SI-1 EMPLACE on 4 gpus run:

```python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=1 --pos=31 --neg=375 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear```
