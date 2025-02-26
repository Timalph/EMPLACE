### Train all SI setups on ACM-1.1
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=1 --pos=31 --neg=375 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=750 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=1125 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=1500 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=90 --pos=365 --neg=365 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear

### Train ablations
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=1 --pos=31 --neg=375 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=fixed_margin_loss
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=750 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=fixed_margin_loss
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=1125 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=fixed_margin_loss
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=1500 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=fixed_margin_loss
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=90 --pos=365 --neg=365 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=1 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=fixed_margin_loss

python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=1 --pos=31 --neg=375 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=0 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=750 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=0 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=1125 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=0 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=275 --pos=475 --neg=1500 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=0 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear
python ACM_main.py --multigpu=4 --input_dir=antenna_700_210_train --pos_prox=90 --pos=365 --neg=365 --total_epochs=200 --amount_of_clusters=10000 --image_size 210 700 --patch_size=14 --batch_size=8 --adjusted_batch_size=64 --cutandflip=0 --encoder_architecture=DINODINO14 --grad_clip=.5 --temp_congruent=1 --loss=linear

#Run order prediction for Table 3 and Table 4
python SI_testing.py

#Run finetuning and zeroshot on Ams-Buildings and Ams-Trees - Table 4

#Process AMS-Buildings and AMS-Trees datasets
for VAR in {0..10}; do
    python process_building_dataset.py --set_type=equirectangular_boom --split 70 20 10 --seed=$VAR
    python process_building_dataset.py --set_type=equirectangular_balanced --split 70 20 10 --seed=$VAR
done



#Run zeroshot 
for WINDOW in 8 7 5 9 10 4; do
    for SEED in 0 1 2 3 4 5 6 7 8 9 10; do

        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED --object=boom

        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED --object=boom

        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED --object=boom

        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=val --multiple_threshold 31 65 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --threshold 30 --seed=$SEED --object=boom
        python compute_threshold_CaseStudy.py --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --citypulse_root='/ivi/xfs/talpher' --window_dim $WINDOW  --split=test --multiple_threshold 31 65 --seed=$SEED --object=boom

    done
done

#Run finetuning
for SEED in 0 1 2 3 4 5 6 7 8 9 10; do
    #vanilla dino
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom'  --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced'  --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    #SI-1
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    #SI-2
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    #SI-3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    #SI-4
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced' --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/best_epoch.pth --headtype=DINODINO14 --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    #resnet
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced'  --headtype=resnet --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom'  --headtype=resnet --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    #clip
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_balanced'  --headtype=clip --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
    CUBLAS_WORKSPACE_CONFIG=:16:8 python ACM_train_tmp.py --ATM_buildings_square='equirectangular_boom'  --headtype=clip --grad_clip=5 --nojitter=1  --freeze=0 --citypulse_root='/ivi/xfs/talpher' --image_size 224 224 --patch_size=14 --batchsize=16 --cutandflip=0 --splits 70 20 10 --vanilla_posembed=1 --stop_epochs=3
done

## See results
python latex_table_to_df.py --table=DINODINO14_latex_ATM_buildings_squareboom__tests.txt
python latex_table_to_df.py --table=DINODINO14_latex_ATM_buildings_square_balanced__tests.txt


#See zeroshot results
python latex_table_to_df.py --table=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss/zeroshot_thresholdscores_test.csv
python latex_table_to_df.py --table=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss/zeroshot_thresholdscores_test.csv
python latex_table_to_df.py --table=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss/zeroshot_thresholdscores_test.csv
python latex_table_to_df.py --table=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss/zeroshot_thresholdscores_test.csv


#Detect Change

#Large Change
python aaai_FindChange.py --window 8 8 --threshold=50 --no_of_results=10000 --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss --optimize_with_disk=1 --build_df=1 --build_comparison_dict=1 

#Small Change
python aaai_FindChange.py --window 2 2 --threshold=50 --no_of_results=10000 --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss --optimize_with_disk=1 --isolated_window=1 --thresh_boundaries=1.1 --adaptive_boundary=1 --build_df=1 --build_comparison_dict=1 

#Visualisation of small change in the appendix
python aaai_FindChange.py --calc_linear_change=4 --underthresh=45 --window 2 2 --threshold=50 --no_of_results=10000 --pretrained=DINODINO14_ATM_continuous_saved_models/img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss --optimize_with_disk=1 --isolated_window=1 --thresh_boundaries=1.1 --adaptive_boundary=1 --build_df=1 --build_comparison_dict=1 
