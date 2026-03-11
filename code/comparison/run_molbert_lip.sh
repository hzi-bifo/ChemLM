for no in $(seq 0 99)
do
	
	python molbert/apps/finetune.py \
   		--test_file /vol/projects/gkallerg/chemlm_2/new_lip/lipschitz_${no}.csv \
    		--mode classification \
    		--output_size 2 \
    		--pretrained_model_path .../bbbp_trained_lipschitz_molbert.ckpt \
    		--label_column mw
done

