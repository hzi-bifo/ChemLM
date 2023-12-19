# example python evaluate.py --dataset "bbbp" --epochs 30 --batch_size 16 --lrate 0.00005 --att_heads 12 --layers 8 --augment_numbers 80 --embs "last" --ft "True" --tokenizer_path '/.../models/trial_tokenizer/' --model_path '/.../models/bbbp_unsup_full_80_all_skf_1023/' --save_path '/.../results_chemlm/'

python evaluate.py --dataset "bace_clf" --epochs 30 --batch_size 16 --lrate 0.00005 --att_heads 16 --layers 8 --augment_number 100 --embs "sum_mean" --ft "True"  --tokenizer_path '' --model_path '' --save_path '' #domain-adapted and optimized model
python evaluate.py --dataset "bace_clf" --epochs 30 --batch_size 16 --lrate 0.00005 --att_heads 12 --layers 12 --augment_number 100 --embs "pooling" --ft "False"  --tokenizer_path '' --model_path '' --save_path '' #domain-adapted
python evaluate.py --dataset "bace_clf" --epochs 30 --batch_size 16 --lrate 0.000001 --att_heads 12 --layers 12 --augment_number -1 --embs "pooling" --ft "False"  --tokenizer_path '' --model_path '' --save_path '' #pretrained

python evaluate.py --dataset "bbbp" --epochs 30 --batch_size 16 --lrate 0.00005 --att_heads 12 --layers 8 --augment_number 80 --embs "last" --ft "True"  --tokenizer_path '' --model_path '' --save_path '' #domain-adapted and optimized model
python evaluate.py --dataset "bbbp" --epochs 30 --batch_size 16 --lrate 0.00005 --att_heads 12 --layers 12 --augment_number 80 --embs "pooling" --ft "False" --tokenizer_path '' --model_path '' --save_path '' #domain-adapted
python evaluate.py --dataset "bbbp" --epochs 30 --batch_size 8 --lrate 0.00005 --att_heads 12 --layers 12 --augment_number -1 --embs "pooling" --ft "False" --tokenizer_path '' --model_path '' --save_path '' #pretrained

python evaluate.py --dataset "clintox" --epochs 30 --batch_size 8 --lrate 0.00005 --att_heads 16 --layers 12 --augment_number 80 --embs "last" --ft "True" --tokenizer_path '' --model_path '' --save_path '' #domain-adapted and optimized model
python evaluate.py --dataset "clintox" --epochs 30 --batch_size 8 --lrate 0.0005 --att_heads 12 --layers 12 --augment_number 80 --embs "pooling" --ft "False" --tokenizer_path '' --model_path '' --save_path '' #domain-adapted
python evaluate.py --dataset "clintox" --epochs 30 --batch_size 16 --lrate 0.00005 --att_heads 12 --layers 12 --augment_number -1 --embs "pooling" --ft "False" --tokenizer_path '' --model_path '' --save_path '' #pretrained

