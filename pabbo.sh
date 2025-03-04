#!/bin/bash -l
#SBATCH --output=evaluate_pabbo_%A.out
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --constraint=milan
##SBATCH --gres=gpu:1


# srun python evaluate_hpob.py --config-name=evaluate_hpob experiment.model=PABBO experiment.expid=PABBO_HPOB7609 experiment.device=cuda eval.eval_max_T=100 data.name=HPOB7609 data.standardize=true data.search_space_id="7609" data.d_x=9 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python evaluate_hpob.py --config-name=evaluate_hpob experiment.model=PABBO experiment.expid=PABBO_HPOB5636 experiment.device=cuda eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python evaluate_hpob.py --config-name=evaluate_hpob experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda eval.eval_max_T=100 data.name=HPOB5636 data.standardize=true data.search_space_id="5636" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" train.x_i_range="[-1, 1]"
# srun python evaluate_hpob.py --config-name=evaluate_hpob experiment.model=PABBO experiment.expid=PABBO_HPOB5859 experiment.device=cuda eval.eval_max_T=100 data.name=HPOB5859 data.standardize=true data.search_space_id="5859" data.d_x=6 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 
# srun python evaluate_hpob.py --config-name=evaluate_hpob experiment.model=PABBO experiment.expid=PABBO_HPOB5971 experiment.device=cuda eval.eval_max_T=100 data.name=HPOB5971 data.standardize=true data.search_space_id="5971" data.d_x=16 data.x_range="[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]" 


## GP1D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D_T_64_old experiment.device=cuda data.name=GP1D data.d_x=1 data.x_range="[[-1,1]]" data.max_num_ctx=50 data.min_num_ctx=1 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0 eval.sobol_grid=true

## GP2D
srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_T_64_old experiment.device=cpu data.name=GP2D data.d_x=2 data.x_range="[[-1,1],[-1,1]]" data.max_num_ctx=100 data.min_num_ctx=1 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0 eval.sobol_grid=true

## candy
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=candy data.d_x=2 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=0 eval.plot_seed_id=-1
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=candy data.d_x=2 eval.eval_num_query_points=512 eval.plot_freq=10 eval.plot_dataset_id=0 eval.plot_seed_id=-1
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=candy data.d_x=2 eval.eval_num_query_points=1024 eval.plot_freq=10 eval.plot_dataset_id=0 eval.plot_seed_id=-1

# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_T_64_old experiment.device=cuda eval.eval_max_T=100 data.name=candy data.d_x=2 eval.num_parallel=5 eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=0 eval.plot_seed_id=-1



## sushi
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=sushi data.d_x=4 eval.eval_num_query_points=256 
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=sushi data.d_x=4 eval.eval_num_query_points=512 
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=sushi data.d_x=4 eval.eval_num_query_points=1024

# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cuda eval.eval_max_T=100 data.name=sushi data.d_x=4 eval.eval_num_query_points=256 eval.num_parallel=5 

# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old data.d_x=4 eval.eval_num_query_points=256 
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old data.d_x=4 eval.eval_num_query_points=512 
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cpu eval.eval_max_T=100 data.name=sushi_old data.d_x=4 eval.eval_num_query_points=1024

# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP4D_T_64_old experiment.device=cuda eval.eval_max_T=100 data.name=sushi_old data.d_x=4 eval.eval_num_query_points=256 eval.num_parallel=5 

## forrester1D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D_T_64_old experiment.device=cpu data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]" eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP1D_240225 experiment.device=cpu data.name=forrester1D data.d_x=1 data.x_range="[[0, 1]]" data.Xopt="[[0.75724876]]" data.yopt="[[-6.020740]]" eval.eval_num_query_points=512 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0
# ## branin2D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_240225 experiment.device=cpu data.name=branin2D data.d_x=2 data.x_range="[[-5, 10], [0, 15]]" data.Xopt="[[-3.142,12.275],[3.142,2.275],[9.42478,2.475]]" data.yopt="[[0.397887],[0.397887],[0.397887]]" eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_240225 experiment.device=cpu data.name=branin2D data.d_x=2 data.x_range="[[-5, 10], [0, 15]]" data.Xopt="[[-3.142,12.275],[3.142,2.275],[9.42478,2.475]]" data.yopt="[[0.397887],[0.397887],[0.397887]]" eval.eval_num_query_points=512 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0

# ## beale2D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_240225 experiment.device=cpu data.name=beale2D data.d_x=2 data.x_range="[[-4.5, 4.5], [-4.5, 4.5]]" data.Xopt="[[3, 0.5]]" data.yopt="[[0.]]" eval.eval_num_query_points=256 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP2D_240225 experiment.device=cpu data.name=beale2D data.d_x=2 data.x_range="[[-4.5, 4.5], [-4.5, 4.5]]" data.Xopt="[[3, 0.5]]" data.yopt="[[0.]]" eval.eval_num_query_points=512 eval.plot_freq=10 eval.plot_dataset_id=-1 eval.plot_seed_id=0

## ackley6D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.sobol_grid=true

# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=512 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=1024 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=ackley6D_old data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=256 eval.num_parallel=5 eval.eval_max_T=60 eval.sobol_grid=true

# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=ackley6D data.d_x=6 data.x_range="[[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.sobol_grid=true
## hartmann6D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=512 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=1024 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.num_parallel=5  eval.sobol_grid=true

## hartmann6D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=512 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=1024 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=hartmann6D_old data.d_x=6 data.x_range="[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" data.Xopt="[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]" data.yopt="[[-3.32237]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.num_parallel=5  eval.sobol_grid=true

## rastrigin6D
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rastrigin6D data.d_x=6 data.x_range="[[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rastrigin6D data.d_x=6 data.x_range="[[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=512 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rastrigin6D data.d_x=6 data.x_range="[[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=1024 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rastrigin6D data.d_x=6 data.x_range="[[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]]" data.yopt="[[0.0]]" data.Xopt="[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.num_parallel=5 eval.sobol_grid=true


## levy
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=levy6D data.d_x=6 data.x_range="[[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60  eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=levy6D data.d_x=6 data.x_range="[[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=512 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=levy6D data.d_x=6 data.x_range="[[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=1024 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=levy6D data.d_x=6 data.x_range="[[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10],[-10, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.num_parallel=5 eval.sobol_grid=true

## rosenbrock
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=512 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_gp.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=1024 eval.eval_max_T=60 eval.sobol_grid=true
# srun python evaluate_continuous_parallel.py --config-name=evaluate experiment.model=PABBO experiment.expid=PABBO_GP6D experiment.device=cuda data.name=rosenbrock6D data.d_x=6 data.x_range="[[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10],[-5, 10]]" data.yopt="[[0.0]]" data.Xopt="[[1.0,1.0,1.0,1.0,1.0,1.0]]" eval.eval_num_query_points=256 eval.eval_max_T=60  eval.num_parallel=5 eval.sobol_grid=true
