# ASE-RP

Adaptive Semantic Enhancement with Rank-Preserving Projection for next-POI recommendation.

```bash
python -m pip install -r requirements.txt
python run.py --lr 1e-3 --save_model checkpoints/base.pt
python semantic_teacher.py
python run.py --experiment ase_rp --load_model checkpoints/base.pt \
  --save_model checkpoints/ase_rp.pt --epochs 30 --batch 512 \
  --lr 1e-4 --weight_decay 1e-4 --patience 3 --early_stop 8 --num_workers 0
```

Set `LLM_API_KEY` before running `semantic_teacher.py`.
