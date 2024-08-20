python main.py --dataname adult --method vae --mode train --gpu 0
python main.py --dataname adult --method tabsyn --mode train --gpu 0
python main.py --dataname adult --method tabsyn --mode sample --save_path /rdf/experiments/fair-tab-diffusion-exps/adult/tabsyn/best/synthesis/2024/d_syn.csv
python main.py --dataname adult --method tabsyn --mode sample --save_path /rdf/experiments/fair-tab-diffusion-exps/adult/tabsyn/best/synthesis/2025/d_syn.csv
python main.py --dataname adult --method tabsyn --mode sample --save_path /rdf/experiments/fair-tab-diffusion-exps/adult/tabsyn/best/synthesis/2026/d_syn.csv
