# Replicate experiments
## Figure 3
### GP1D
`srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D experiment.device=cuda data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0`

`srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$i experiment.model=qTS experiment.device=cpu data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1`

`srun python baseline.py --config-name=evaluate eval.dataset_id=0 eval.seed_id=$i experiment.model=mpes experiment.device=cpu data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]"`
### GP2D

### Forrester1D
`srun python evaluate_test_function.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D experiment.device=cuda data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]" eval.plot_freq=10`


### Branin2D

`data.name=branin2D data.d_x=2 data.x_range="[[-5, 10], [0, 15]]" data.Xopt="[[-3.142,12.275],[3.142,2.275],[9.42478,2.475]]" data.yopt="[[0.397887],[0.397887],[0.397887]]"`

### Beale2D

`data.name=beale2D data.d_x=2 data.x_range="[[-4.5, 4.5], [-4.5, 4.5]]" data.Xopt="[[3, 0.5]]" data.yopt="[[0.]]"`


### Ackley6D

`data.name=ackley6D data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]"`


`data.name=hartmann6D data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]"`


### HPOB