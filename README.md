# Replicate experiments
## Figure 3
### GP1D
`srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D experiment.device=cuda data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0`

`srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$i experiment.model=qTS experiment.expid=qTS_GP1D experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1`

`srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$i experiment.model=mpes experiment.expid=qTS_GP1D experiment.device=cpu data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]"`
### GP2D

### Forrester1D
`srun python evaluate_test_function.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D experiment.device=cuda data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]" eval.plot_freq=10`

### Branin2D

### Beale2D



