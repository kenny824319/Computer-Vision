# Prerequisite
1\. put p2_data in part2
2\. In cfg.py 'model_type' should be 'ResNext50'
# Quick Start
First, run main.py to get the best_model.pt, which will in ./Model/ResNext50
```sh
cd part2
python main.py
```
Second, simply run eval.py
```sh
python eval.py --model ResNext50
```