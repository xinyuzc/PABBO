import torch
import torch.nn as nn
from torch import Tensor
import os
import os.path as osp
import time
import wandb

from omegaconf import DictConfig
import hydra
from wandb_wrapper import init as wandb_init, save_artifact
from policies.transformer import TransformerModel
from data.sampler import OptimizationSampler
from data.environment import generate_pair_set, sample_random_pairs_from_Q
from data.utils import *
from data.hpob import *
from policy_learning import *
from utils.paths import RESULT_PATH, DATASETS_PATH
from utils.log import get_logger, Averager
from utils.losses import preference_cls_loss, accuracy, kendalltau_correlation


@hydra.main(version_base=None, config_path="configs")
def main(config: DictConfig):
    # experiment directory
    exp_path = osp.join(
        os.getcwd(), RESULT_PATH, config.experiment.model, config.experiment.expid
    )

    # wandb files will be saved under experiment directory
    if config.experiment.wandb:
        wandb_init(config=config, **config.wandb, dir=exp_path)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(config.experiment.device)

    model = TransformerModel(**config.model)
    train(
        expid=config.experiment.expid,
        exp_path=exp_path,
        model=model,
        dataname=config.data.name,
        x_range=config.data.x_range,
        sampler_kwargs=config.train.sampler,
        min_num_ctx=config.data.min_num_ctx,
        max_num_ctx=config.data.max_num_ctx,
        standardize=config.data.standardize,
        search_space_id=config.data.search_space_id,
        p_noise=config.train.p_noise,
        n_steps=config.train.n_steps,
        n_burnin=config.train.n_burnin,
        batch_size=config.train.train_batch_size,
        batch_size_2=config.train.ac_train_batch_size,
        lr=config.train.lr,
        lr_2=config.train.ac_lr,
        n_random_pairs=config.train.n_random_pairs,
        num_prediction_points=config.train.num_prediction_points,
        num_query_points=config.train.num_query_points,
        num_init_pairs=config.train.num_init_pairs,
        n_trajectories=config.train.n_trajectories,
        max_T=config.train.max_T,
        discount_factor=config.train.discount_factor,
        regret_option=config.train.regret_option,
        loss_weight=config.train.loss_weight,
        auxiliary_ratio=config.train.auxiliary_ratio,
        print_freq=config.train.print_freq,
        eval_freq=config.train.eval_freq,
        save_freq=config.train.save_freq,
        ranking_reward=config.train.ranking_reward,
        sobol_grid=config.train.sobol_grid,
        seed=config.train.train_seed,
        resume=config.experiment.resume,
        device=config.experiment.device,
        wb=config.experiment.wandb,
    )


def replicate_batch(
    batch: Tensor,  # [B, N, D]
    num_replicas: int,
) -> Tensor:  # [B * num_replicas, N, D]
    """replicate the batch for `num_replicas` times"""
    B, N, D = batch.shape
    batch = batch.unsqueeze(1)  # [B, 1, N, D]
    batch = batch.expand(-1, num_replicas, -1, -1)  # [B, num_replicas, N, D]
    batch = batch.reshape(-1, N, D)  # [B * num_replicas, N, D]
    return batch


def train(
    expid: str,
    exp_path: str,
    model: TransformerModel,
    dataname: str,
    x_range: list,  # [d_x, 2]
    sampler_kwargs: dict,
    min_num_ctx: int,
    max_num_ctx: int,
    standardize: bool = False,
    search_space_id: str = "",
    p_noise: float = 0.0,
    n_steps: int = 10000,
    n_burnin: int = 3000,
    batch_size: int = 128,
    batch_size_2: int = 16,
    lr: float = 1e-3,
    lr_2: float = 3e-5,
    n_random_pairs: int = 100,
    num_prediction_points: int = 100,
    num_query_points: int = 100,
    num_init_pairs: int = 0,
    n_trajectories: int = 20,
    max_T: int = 64,
    discount_factor: float = 0.98,
    regret_option: str = "simple_regret",
    loss_weight: float = 1.0,
    auxiliary_ratio: float = 1.0,
    print_freq: int = 200,
    eval_freq: int = 1000,
    save_freq: int = 500,
    ranking_reward: bool = False,
    sobol_grid: bool = False,
    seed: int = 0,
    resume: bool = False,
    device: str = "cuda",
    wb: bool = True,  # whether to use wandb
):
    n_gpus = torch.cuda.device_count()  # the number of gpus

    # checkpoint
    if osp.exists(exp_path + "/ckpt.tar"):
        if not resume:
            raise FileExistsError(exp_path)
    else:
        os.makedirs(exp_path, exist_ok=True)

    logfilename = os.path.join(
        exp_path, f'train_{dataname}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    )
    logger = get_logger(file_name=logfilename, mode="w")

    set_all_seeds(seed)

    # define data sampler
    if dataname.startswith("GP"):
        # GP synthesis data
        sampler = OptimizationSampler(
            device=device,
            **sampler_kwargs,
        )
    elif dataname.startswith("HPOB"):
        # Hyperparameter optimization data
        sampler = HPOBHandler(
            root_dir=osp.join(
                hydra.utils.get_original_cwd(), DATASETS_PATH, "hpob-data"
            ),
            mode="v3-train-augmented",
        )
    else:
        raise NotImplementedError(f"Dataset {dataname} not implemented.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # load checkpoint
    if resume:
        ckpt = torch.load(os.path.join(exp_path, "ckpt.tar"), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        expdir = ckpt["expdir"]
        start_step = ckpt["step"]
        if start_step == n_burnin + 1:
            # using smaller learning rate when introducing acquisition task
            for g in optimizer.param_groups:
                g["lr"] = lr_2

    else:
        expdir = os.getcwd()
        start_step = 1

    if not resume:
        logger.info(f"Experiment: {expid}")
        logger.info(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n"
        )
        logger.info(f"Train on {dataname}.")

    # simple DataParallel
    if n_gpus > 1:
        print(f"Model is copied and distributed to {n_gpus} GPUs")
        model = nn.DataParallel(model)
        model.to(device)

    # record average metrics
    ravg = Averager()

    d_x = len(x_range)

    assert (
        n_random_pairs >= max_T
    ), f"n_random_pairs {n_random_pairs} should be no less than max_T {max_T}"

    logger.info(f"Number of prediction points: {num_prediction_points}")
    logger.info(f"Number of query points: {num_query_points}")
    logger.info(f"Number of random pairs: {n_random_pairs}")

    for epoch in range(start_step, n_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # create datasets
        if epoch <= n_burnin:
            B = batch_size
            # during burnin phase, only sample the prediction dataset
            X_pred, y_pred, _, _ = sampler.sample(
                batch_size=B,
                max_num_ctx_points=max_num_ctx,
                num_total_points=num_prediction_points,
                x_range=x_range,  # [d_x, 2]
                sobol_grid=sobol_grid,
                evaluate=False,
                search_space_id=str(search_space_id),
                standardize=standardize,
                train_split="pred",
                split="train",
                device=device,
            )
        else:
            B = batch_size_2
            # sample the prediction and query dataset
            if dataname.startswith("GP"):
                X, y, Xopt, yopt = sampler.sample(
                    batch_size=B,
                    max_num_ctx_points=max_num_ctx,
                    num_total_points=num_prediction_points + num_query_points,
                    x_range=x_range,  # [d_x, 2]
                    sobol_grid=sobol_grid,
                    evaluate=False,
                )

                X_pred = X[:, :-num_query_points]
                y_pred = y[:, :-num_query_points]
                X_ac = X[:, -num_query_points:]
                y_ac = y[:, -num_query_points:]
            else:
                # HPOB data: sample prediction and query datasets from different split to prevent information leak
                X_pred, y_pred, _, _ = sampler.sample(
                    batch_size=B,
                    num_total_points=num_prediction_points,
                    search_space_id=str(search_space_id),
                    standardize=standardize,
                    train_split="pred",
                    split="train",
                    device=device,
                )
                X_ac, y_ac, Xopt, yopt = sampler.sample(
                    num_total_points=num_query_points,
                    batch_size=B,
                    search_space_id=str(search_space_id),
                    standardize=standardize,
                    train_split="acq",
                    split="train",
                    device=device,
                )

            # reuse the query dataset for `num_trajectories` times for stability: [Bt, N, d_x]...
            X_ac = replicate_batch(X_ac, n_trajectories)
            y_ac = replicate_batch(y_ac, n_trajectories)
            Xopt = replicate_batch(Xopt, n_trajectories)
            yopt = replicate_batch(yopt, n_trajectories)

            query_pair_idx, query_pair, query_pair_y, query_c = generate_pair_set(
                X=X_ac,
                y=y_ac,
                num_total_points=num_query_points,
                n_random_pairs=n_random_pairs,
                p_noise=p_noise,
                ranking_reward=ranking_reward,
            )

            Bt, num_query_pairs, _ = query_pair.shape

            # expand query_pair_idx
            assert query_pair_idx.shape == (
                num_query_pairs,
                2,
            ), f"query_pair_idx of shape {query_pair_idx.shape} should be of shape ({num_query_pairs}, 2)."

            # [num_query_pairs, 2] -> [num_query_pairs, n_gpus, 2] -> [num_query_pairs * n_gpus, 2]
            query_pair_idx_tiled = query_pair_idx.tile(n_gpus, 1)
            # mask to register queried pairs: [Bt, num_query_pairs, 1]
            mask = torch.ones((Bt, num_query_pairs, 1)).bool()

            # sample random starting pairs from Q
            (context_pairs, context_pairs_y, context_c, init_pair_idx) = (
                sample_random_pairs_from_Q(
                    pair=query_pair,
                    pair_y=query_pair_y,
                    c=query_c,
                    num_samples=num_init_pairs,
                )
            )
            mask[:, init_pair_idx] = 0.0  # NOTE mask out sampled ones

            # policy learning setup
            rewards = []
            entropys = []
            log_probs = []
            rewards = []

        # generate prediction set: [B, n_random_pairs, 2 * d_x], ...
        _, src_pairs, src_pairs_y, src_c = generate_pair_set(
            X=X_pred,
            y=y_pred,
            num_total_points=num_prediction_points,
            n_random_pairs=n_random_pairs,
            p_noise=p_noise,
        )

        # free up memory
        X_pred = X_pred.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        del X_pred
        del y_pred

        cls_losses = 0.0

        # optimization loop with T time budget
        for t in range(1, max_T + 2):
            # prediction task: randomly split the prediction set into context and target
            _, ctx_idx, _, tar_idx = get_random_split_data(
                total_num=num_prediction_points,
                min_num_ctx=min_num_ctx,
                max_num_ctx=max_num_ctx,
            )

            # predicted function values
            pred_tar_f = model(
                query_src=src_pairs[:, ctx_idx],
                c_src=src_c[:, ctx_idx],
                eval_pos=src_pairs[:, tar_idx].view(
                    len(src_pairs), -1, d_x
                ),  # flatten the target pairs into locations
                acquire=False,
            )[1]

            # compute and record BCE loss
            cls_loss = preference_cls_loss(
                f=pred_tar_f, c=src_c[:, tar_idx], reduction="mean"
            )
            cls_losses += cls_loss

            # record accuracy and kendall tau correlation
            acc = accuracy(f=pred_tar_f.detach(), c=src_c[:, tar_idx], reduce=True)
            kt_cor = kendalltau_correlation(
                pred_tar_f.detach(),
                src_pairs_y[:, tar_idx].view(len(src_pairs_y), -1, 1),
                reduce=True,
            )
            ravg.update("acc", acc)
            ravg.update("kt_cor", kt_cor)

            # optimization task
            if epoch > n_burnin:
                # take action / suggest next query pair
                acq_values, next_pair_idx, log_prob, entropy = action(
                    model=model,
                    context_pairs=context_pairs,
                    context_preference=context_c,
                    t=t,
                    T=max_T,
                    X_pending=X_ac,
                    pair_idx=query_pair_idx_tiled,  # NOTE DataParallel will split every tensor along the first dim
                    mask=mask,
                )

                # compute and record reward for previous step
                if t > 1:
                    reward = get_reward(
                        context_pairs_y=context_pairs_y,
                        acq_values=acq_values,
                        pair_y=query_pair_y,
                        regret_option=regret_option,
                    )
                    rewards.append(reward)
                    if t > max_T:
                        break

                # update observations and candidate query pair set
                next_pair_idx = next_pair_idx[:, None, None]  # [Bt, 1, 1]

                # mask out the selected query pair
                mask.scatter_(
                    dim=1,
                    index=next_pair_idx,
                    src=torch.zeros_like(mask, device=mask.device),
                )

                next_pair = gather_data_at_index(data=query_pair, idx=next_pair_idx)
                # only for reward computation
                next_pair_y = gather_data_at_index(data=query_pair_y, idx=next_pair_idx)
                next_c = gather_data_at_index(data=query_c, idx=next_pair_idx)

                # update observations
                context_pairs = torch.cat((context_pairs, next_pair), dim=1)
                context_c = torch.cat((context_c, next_c), dim=1)
                context_pairs_y = torch.cat((context_pairs_y, next_pair_y), dim=1)

                # record log probabilities and entropy
                log_probs.append(log_prob)
                entropys.append(entropy)

        # averaged BCE loss
        cls_loss = cls_losses / max_T

        # compute the policy learning loss given trajectories information
        if epoch > n_burnin:
            # best utility value of the context pairs; only for computing reward
            y_best_acq = torch.max(
                context_pairs_y.flatten(start_dim=1), dim=-1
            ).values.squeeze()

            # best utility value of the entire query pair set
            y_opt = torch.max(query_pair_y.flatten(start_dim=1), dim=-1).values

            # simple regret: difference between the best possible and the best observed utility value
            final_simple_regret = (y_opt - y_best_acq).mean().item()

            # compute the policy learning loss
            policy_loss, entropy = finish_episode(
                entropys=entropys,
                rewards=rewards,
                log_probs=log_probs,
                discount_factor=discount_factor,
                sum_over_tra=True,  # sum loss over trajectories or mean
            )

            # auxiliary ratio: randomly include prediction task loss for ablation. Default 1.0
            lw = loss_weight * (auxiliary_ratio > torch.rand((1), device=device))
            loss = policy_loss + lw * cls_loss
        else:
            policy_loss, final_simple_regret, entropy = (
                0.0,
                0.0,
                0.0,
            )
            loss = cls_loss

        # backpropogation
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()

        # record metrics
        ravg.update("loss", loss)
        ravg.update("cls_loss", cls_loss)
        ravg.update("policy_loss", policy_loss)
        ravg.update("final_simple_regret", final_simple_regret)
        ravg.update("entropy", entropy)

        if epoch % print_freq == 0:
            line = f"{expid} step {epoch} "
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)
            ravg.reset()

        if wb:
            wandb.log(
                {
                    "entropy": entropy,
                    "final_simple_regret": final_simple_regret,
                    "acc": acc,
                    "kt_cor": kt_cor,
                    "loss": loss,
                    "cls_loss": cls_loss,
                    "policy_loss": policy_loss,
                }
            )

        # saving checkpoints
        if epoch % save_freq == 0 or epoch == n_steps:
            # save the model so you can load it to any device
            model_state_dict = (
                model.state_dict() if n_gpus <= 1 else model.module.state_dict()
            )
            ckpt = {
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "expdir": expdir,
                "step": epoch + 1,
            }

            # save to the result saving path
            torch.save(ckpt, os.path.join(exp_path, f"ckpt.tar"))

            # save to wandb if used
            if wb:
                save_artifact(
                    run=wandb.run,
                    local_path=os.path.join(exp_path, f"ckpt.tar"),
                    name="checkpoint",
                    type="model",
                )

        # using smaller learning rate when introducing acquisition task
        if epoch == n_burnin:
            for g in optimizer.param_groups:
                g["lr"] = lr_2

        epoch += 1


if __name__ == "__main__":
    main()
