# finetune
python fine_tune_avatar_plus.py --subject-idx 313 --num-workers 8 --subsampling-rate 1 \
--optim-epochs 256 --epochs-per-run 10000 --exclude-hand --use-normal --exp-suffix _313_sample1_e256_eh_un \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
python fine_tune_avatar_plus.py --subject-idx 315 --num-workers 8 --subsampling-rate 1 \
--optim-epochs 256 --epochs-per-run 10000 --exclude-hand --use-normal --exp-suffix _315_sample1_e256_eh_un \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
# inference
python inference.py --high-res --subject-idx 313 --subsampling-rate 5 --exp-suffix _313_sample1_e256_eh_un \
--extrapolation --aist-sequence gBR_sBM_cAll_d04_mBR1_ch07 \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
python inference.py --high-res --subject-idx 313 --subsampling-rate 5 --exp-suffix _313_sample1_e256_eh_un \
--extrapolation --aist-sequence gLO_sBM_cAll_d13_mLO0_ch10 \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
python inference.py --high-res --subject-idx 315 --subsampling-rate 5 --exp-suffix _315_sample1_e256_eh_un \
--extrapolation --aist-sequence gLO_sBM_cAll_d13_mLO0_ch10 \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
python inference.py --high-res --subject-idx 315 --subsampling-rate 5 --exp-suffix _315_sample1_e256_eh_un \
--extrapolation --aist-sequence gBR_sBM_cAll_d04_mBR1_ch07 \
configs/meta-avatar/conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus.yaml
# render
#python render.py --exp-suffix _313_sample1_e128 --subject-idx 313