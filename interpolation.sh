# finetune
#python fine_tune_avatar_plus.py --subject-idx 313 --num-workers 8 --subsampling-rate 2 \
#--optim-epochs 128 --epochs-per-run 10000 --exclude-hand --use-normal --exp-suffix _313_sample2_e32_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python fine_tune_avatar_plus.py --subject-idx 313 --num-workers 8 --subsampling-rate 2 \
#--optim-epochs 128 --epochs-per-run 10000 --exclude-hand --use-normal --exp-suffix _313_sample2_e64_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
# inference
python inference.py --high-res --subject-idx 313 --subsampling-rate 2 --start-offset 0 --exp-suffix _313_sample2_e32_eh_un \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python inference.py --high-res --subject-idx 313 --subsampling-rate 2 --start-offset 1 --exp-suffix _313_sample2_e64_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
# evaluation
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 0 --subject-idx 313 \
#--bi-directional --high-res --exp-suffix _313_sample2_e128_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 1 --subject-idx 313 \
#--bi-directional --high-res --exp-suffix _313_sample2_e128_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 0 --subject-idx 315 \
#--bi-directional --high-res --exp-suffix _315_sample2_e128_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 1 --subject-idx 315 \
#--bi-directional --high-res --exp-suffix _315_sample2_e128_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 0 --subject-idx 313 \
#--bi-directional --high-res --exp-suffix _313_sample2_e128_eh \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 1 --subject-idx 313 \
#--bi-directional --high-res --exp-suffix _313_sample2_e128_eh \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 0 --subject-idx 315 \
#--bi-directional --high-res --exp-suffix _315_sample2_e128_eh \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 1 --subject-idx 315 \
#--bi-directional --high-res --exp-suffix _315_sample2_e128_eh \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 0 --subject-idx 313 \
--bi-directional --high-res --exp-suffix _313_sample2_e32_eh_un \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
#python evaluation/eval_plus.py --subsampling-rate 2 --start-offset 1 --subject-idx 313 \
#--bi-directional --high-res --exp-suffix _313_sample2_e64_eh_un \
#configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml