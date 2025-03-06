# Overview
This is the code repository for paper PABBO: Preferential Amortized Black-Box Optimization accepted to ICLR 2025 by Xinyu Zhang, Daolang Huang, Samuel Kaski, and Julien Martinelli. 

# Setup 
1. Install dependencies using the requirements.txt: 
    ```bash
    pip install -r requirements
    ```
2. Create `.env` file and insert your W&B API key:
    ```
    WANDB_API_KEY=...
    ```
3. Save HPOB and real-world data under following paths: 
   1. Candy: `datasets/candy-data/candy-data.csv`
   2. Sushi: `datasets/sushi-data/sushi3.idata`, `datasets/sushi-data/sushi3b.5000.10.score`
   3. HPOB: `datasets/hpob-data/meta-test-dataset.json`, `datasets/hpob-data/meta-validation-dataset.json`, `datasets/hpob-data/meta-train-dataset-augmented.json`
# Training 
Configuration is saved under path `configs/evaluate.yaml`. Adjust and run the scripts. 

Checkpoint will be saved under path `results/evaluation/PABBO/{experiment.expid}`.

To train PABBO on **1-dimensional GP-based samples**: 
```bash
python train.py --config-name=train \
experiment.expid=PABBO_GP1D \
data.name=GP1D \
data.d_x=1 \
data.x_range="[[-1,1]]" \
data.min_num_ctx=1 \
data.max_num_ctx=50 \
train.num_query_points=100 \
train.num_prediction_points=100 \
train.n_random_pairs=100 
```
To train PABBO on **2-dimensional GP-based samples**:
```bash
python train.py --config-name=train  \
experiment.expid=PABBO_GP2D  \
data.name=GP2D  \
data.d_x=2 \
data.x_range="[[-1,1], [-1, 1]]"  \
data.min_num_ctx=1  \
data.max_num_ctx=100 \
train.num_query_points=200 \
train.num_prediction_points=200 \
train.n_random_pairs=200 
```
To train PABBO on **4-dimensional GP-based samples**:
```bash
python train.py --config-name=train  \
experiment.expid=PABBO_GP4D \
data.name=GP4D \
data.d_x=4 \
data.x_range="[[-1, 1],[-1, 1],[-1, 1],[-1, 1]]" \
data.min_num_ctx=1 \
data.max_num_ctx=100 \
train.num_query_points=300 \
train.num_prediction_points=300 \
train.n_random_pairs=300 
```
To train PABBO on **6-dimensional GP-based samples**:
```bash
python train.py --config-name=train  \
experiment.expid=PABBO_GP6D_st \
data.name=GP6D \
data.d_x=6 \
data.x_range="[[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1]]" \
data.min_num_ctx=1 \
data.max_num_ctx=200 \
train.num_query_points=300 \
train.num_prediction_points=300 \
train.n_random_pairs=300 
```
To train PABBO on **HPOB5971**:
```bash
python train.py --config-name=train \
experiment.expid=PABBO_HPOB5971 \
data.name=HPOB5971 \
data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]"  \
data.standardize=true \
data.min_num_ctx=1 \
data.max_num_ctx=200 \
data.search_space_id=5971 \
data.d_x=16 \
train.num_query_points=300 \
train.num_prediction_points=300 \
train.n_random_pairs=300 
```
To train PABBO on **HPOB7609**:
```bash
python train.py --config-name=train
experiment.expid=PABBO_HPOB7609 \
data.name=HPOB7609 \
data.standardize=true \
data.search_space_id=7609 \
data.d_x=9 \
data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" \
data.min_num_ctx=1 \
data.max_num_ctx=200 \
train.num_query_points=300 \
train.num_prediction_points=300 \
train.n_random_pairs=300 
```
To train PABBO on **HPOB5636**:
```bash
python train.py --config-name=train \
experiment.expid=PABBO_HPOB5636 \
data.name=HPOB5636 \
data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" \ 
data.standardize=true \
data.min_num_ctx=1 \
data.max_num_ctx=200 \
data.search_space_id=5636 \
data.d_x=6 \
train.num_query_points=300 \
train.num_prediction_points=300 \
train.n_random_pairs=300 
```
To train PABBO on **HPOB5859**:
```bash
python train.py --config-name=train \
experiment.expid=PABBO_HPOB5859 \
data.name=HPOB5859 \
data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]"  \
data.standardize=true \
data.min_num_ctx=1 \
data.max_num_ctx=200 \
data.search_space_id=5859 \
data.d_x=6 \
train.num_query_points=300 \
train.num_prediction_points=300 \
train.n_random_pairs=300 
```
# Testing 
Configuration is saved under path `configs/evaluate.yaml`. Run the scripts with test data information passed as arguments. 

Results will be saved under path `results/evaluation/{data.name}/PABBO/{experiment.expid}`.

To test PABBO on **1-dimensional GP-based samples**:

```bash 
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP1D  \
experiment.device=cpu  \
data.name=GP1D  \
data.d_x=1  \
data.x_range="[[-1,1]]"  \
data.max_num_ctx=50  \
data.min_num_ctx=1  \
eval.eval_num_query_points=256  \
eval.plot_freq=10 \
 eval.plot_dataset_id=-1  \
 eval.plot_seed_id=0  \
 eval.sobol_grid=true \
```


To test PABBO on **2-dimensional GP-based samples**: 
```bash
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP2D  \
experiment.device=cpu  \
data.name=GP2D  \
data.d_x=2  \
data.x_range="[[-1,1],[-1,1]]"  \
data.max_num_ctx=100  \
data.min_num_ctx=1  \
eval.eval_num_query_points=256  \
eval.plot_freq=10  \
eval.plot_dataset_id=-1  \
eval.plot_seed_id=0  \
eval.sobol_grid=true 
```
To test PABBO on **forrester1D function**: 
```bash 
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP1D  \
experiment.device=cpu  \
eval.eval_max_T=30 \
eval.eval_num_query_points=256  \
eval.num_parallel=1 \
eval.plot_freq=10  \
eval.plot_dataset_id=0  \
eval.plot_seed_id=-1 \
data.name=forrester1D \
data.d_x=1  \
data.x_range="[[0, 1]]" \
data.Xopt="[[0.75724876]]"  \
data.yopt="[[-6.020740]]" 

```

To test PABBO on **branin2D function**: 
```bash 
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP2D  \
experiment.device=cpu  \
eval.eval_max_T=30 \
eval.eval_num_query_points=256  \
eval.num_parallel=1 \
eval.plot_freq=10  \
eval.plot_dataset_id=0  \
eval.plot_seed_id=-1 \
data.name=branin2D  \
data.d_x=2  \
data.x_range="[[-5, 10], [0, 15]]"  \
data.Xopt="[[-3.142,12.275],[3.142,2.275],[9.42478,2.475]]" 
data.yopt="[[0.397887],[0.397887],[0.397887]]" 
```
To test PABBO on **beale2D function**: 
```bash 
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP2D  \
experiment.device=cpu  \
eval.eval_max_T=30 \
eval.eval_num_query_points=256  \
eval.num_parallel=1 \
eval.plot_freq=10  \
eval.plot_dataset_id=0  \
eval.plot_seed_id=-1 \
data.name=beale2D  \
data.d_x=2  \
data.x_range="[[-4.5, 4.5], [-4.5, 4.5]]"  \
data.Xopt="[[3, 0.5]]"  \
data.yopt="[[0.]]" 
```
To test PABBO on **ackley6D function**: 
```bash
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP6D  \
experiment.device=cpu  \
eval.eval_max_T=60 \
eval.eval_num_query_points=512 \
eval.num_parallel=1 \
data.name=ackley6D  \
data.d_x=6  \
data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]"  \
data.yopt="[[0.0]]"  \
data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" 
```
To test PABBO on **hartmann6D function**: 
```bash
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP6D  \
experiment.device=cpu  \
eval.eval_max_T=60 \
eval.eval_num_query_points=512 \
eval.num_parallel=1 \
data.name=hartmann6D  \
data.d_x=6  \
data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]"  \
data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]"  \
data.yopt="[[-3.32237]]" 
```
To test PABBO on **levy6D function**: 
```bash
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP6D  \
experiment.device=cpu  \
eval.eval_max_T=60 \
eval.eval_num_query_points=512 \
eval.num_parallel=1 \
data.name=levy6D \
data.d_x=6 \
data.x_range="[[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10]]" \
data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" \
data.yopt="[[0.0]]" 
```
To test PABBO on **HPOB5859**: 
```bash
python evaluate_hpob.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_HPOB5859  \
experiment.device=cpu \
train.x_i_range="[0, 1]" \
eval.eval_max_T=100 \
data.name=HPOB5859  \
data.standardize=true  \
data.search_space_id="5859" \
data.d_x=6  \
data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]"  \
```
To test PABBO on **candy**: 
```bash
python evaluate_gp.py --config-name=evaluate 
experiment.model=PABBO \
experiment.expid=PABBO_GP2D \
experiment.device=cpu  \
eval.eval_max_T=100  \
data.name=candy  \
data.d_x=2  \
eval.eval_num_query_points=512  \
eval.plot_freq=10  \
eval.plot_dataset_id=0  \
eval.plot_seed_id=-1 
```
To test PABBO on **sushi**:
```bash 
python evaluate_gp.py --config-name=evaluate  \
experiment.model=PABBO  \
experiment.expid=PABBO_GP4D  \
experiment.device=cpu \
eval.eval_max_T=100  \
data.name=sushi  \
data.d_x=4  \
eval.eval_num_query_points=512  
```
## Baseline
Specify: 
- `experiment.model` in [rs, qTS, qEI, qEUBO, qNEI, mpes]: test acquisition function name.
- `eval.dataset_id`: the id of dataset.
- `seed_id`: the id of random seeds. 

Results will be saved under path `results/evaluation/{data.name}/{experiment.model}/{eval.dataset_id}/metric_{eval.seed_id}.pt`.

To test PBO on **hartmann6D function**: 
```bash
python baseline.py --config-name=evaluate  \
eval.dataset_id=0  \
eval.seed_id=0  \
experiment.wandb=false  \
experiment.model=rs  \
experiment.device=cpu  \
data.name=hartmann6D  \
data.d_x=6  \
data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]"  \
data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]"  \
data.yopt="[[-3.32237]]"  \
eval.eval_max_T=60 
```
To test PBO on **sushi**
```bash
python baseline.py --config-name=evaluate 
eval.dataset_id=0 
eval.seed_id=0 
experiment.wandb=false 
experiment.model=rs 
experiment.device=cpu 
eval.eval_max_T=100 
data.name=sushi 
```
