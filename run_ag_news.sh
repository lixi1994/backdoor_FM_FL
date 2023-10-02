# cross-silo
python3 FL_text.py --epochs 50 --num_users 10 --frac 1 --local_ep 3 --local_bs 32 --pre_lr 2e-5 --lr 1e-5 --model distill_bert --dataset ag_news --gpu --optimizer adamw --iid --mode ours
python3 FL_text.py --epochs 50 --num_users 10 --frac 1 --local_ep 3 --local_bs 32 --pre_lr 2e-5 --lr 1e-5 --model distill_bert --dataset ag_news --gpu --optimizer adamw --mode ours
python3 FL_text.py --epochs 50 --num_users 10 --frac 1 --local_ep 3 --local_bs 32 --pre_lr 2e-5 --lr 1e-5 --model distill_bert --dataset ag_news --gpu --optimizer adamw --iid --mode clean
python3 FL_text.py --epochs 50 --num_users 10 --frac 1 --local_ep 3 --local_bs 32 --pre_lr 2e-5 --lr 1e-5 --model distill_bert --dataset ag_news --gpu --optimizer adamw --mode clean
python3 FL_text.py --epochs 50 --num_users 10 --frac 1 --local_ep 3 --local_bs 32 --pre_lr 2e-5 --lr 1e-5 --model distill_bert --dataset ag_news --gpu --optimizer adamw --iid --mode BD_baseline --attackers 1
python3 FL_text.py --epochs 50 --num_users 10 --frac 1 --local_ep 3 --local_bs 32 --pre_lr 2e-5 --lr 1e-5 --model distill_bert --dataset ag_news --gpu --optimizer adamw --mode BD_baseline --attackers 1