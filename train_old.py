import torch
import torch.nn as nn
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
    with wandb_init(config=config, **config.wandb) as _:
        # set default device and data type
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(config.experiment.device)

        model = TransformerModel(**config.model)
        train(config, model)


def train(config: DictConfig, model: TransformerModel):
    device = torch.device(config.experiment.device)
    n_gpus = torch.cuda.device_count()  # the number of gpus

    # result saving path, `/results/PABBO/expid`
    root = osp.join(
        hydra.utils.get_original_cwd(),
        RESULT_PATH,
        config.experiment.model,
        config.experiment.expid,
    )

    # checkpoint
    if osp.exists(root + "/ckpt.tar"):
        if not config.experiment.resume:
            raise FileExistsError(root)
    else:
        os.makedirs(root, exist_ok=True)

    logfilename = os.path.join(
        root, f'train_{config.data.name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    )
    logger = get_logger(file_name=logfilename, mode="w")

    set_all_seeds(config.train.train_seed)
    if config.data.name.startswith("GP"):
        # sampler for GP-based synthetic functions
        sampler = OptimizationSampler(
            device=config.experiment.device,
            **config.train.sampler,
        )
    elif config.data.name.startswith("HPOB"):
        # sampler for HPOB data
        sampler = HPOBHandler(
            root_dir=osp.join(
                hydra.utils.get_original_cwd(), DATASETS_PATH, "hpob-data"
            ),
            mode="v3-train-augmented",
        )
    else:
        raise NotImplementedError(
            f"Sampler for data {config.data.name} is not implemented."
        )

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.n_steps
    )

    # load saved model if resume
    if config.experiment.resume:
        ckpt = torch.load(os.path.join(root, "ckpt.tar"), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        expdir = ckpt["expdir"]
        start_step = ckpt["step"]
    else:
        expdir = os.getcwd()
        start_step = 1

    # log experiment setup
    if not config.experiment.resume:
        logger.info(f"Experiment: {config.experiment.expid}")
        logger.info(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n"
        )
        logger.info(f"Train on {config.data.name}.")

    # DataParallel
    if n_gpus > 1:
        print(f"Model is copied and distributed to {n_gpus} GPUs")
        model = nn.DataParallel(model)
        model.to(device)

    # averager for recording metrics
    ravg = Averager()

    B = config.train.train_batch_size
    d_x = len(config.data.x_range)

    if config.train.n_random_pairs < config.train.max_T:
        raise ValueError(
            f"The size of candidate query pair set given by `n_random_pairs` should be no less than the number of queries given by `max_T`."
        )
    logger.info(
        f"Number of prediction points: {config.train.num_prediction_points}, number of query points: {config.train.num_prediction_points}, number of pairs in prediction set and query pair set: {config.train.n_random_pairs}"
    )

    # training loops
    for epoch in range(start_step, config.train.n_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # create datasets
        if epoch <= config.train.n_burnin:
            # warmup: only train the prediction head on prediction dataset
            X_pred, y_pred, _, _ = sampler.sample(
                batch_size=B,
                max_num_ctx_points=config.data.max_num_ctx,
                num_total_points=config.train.num_prediction_points,
                x_range=config.data.x_range,  # [d_x, 2]
                sobol_grid=config.train.sobol_grid,
                evaluate=False,
                search_space_id=str(config.data.search_space_id),
                standardize=config.data.standardize,
                train_split="pred",
                split="train",
                device=config.experiment.device,
            )
        else:
            if config.data.name.startswith("GP"):
                # sample both prediction and query dataset
                X, y, Xopt, yopt = sampler.sample(
                    batch_size=B,
                    max_num_ctx_points=config.data.max_num_ctx,
                    num_total_points=config.train.num_prediction_points
                    + config.train.num_query_points,
                    x_range=config.data.x_range,  # [d_x, 2]
                    sobol_grid=config.train.sobol_grid,
                    evaluate=False,
                )

                X_pred = X[:, : -config.train.num_query_points]
                y_pred = y[:, : -config.train.num_query_points]
                X_ac = X[:, -config.train.num_query_points :]
                y_ac = y[:, -config.train.num_query_points :]
            else:
                # for HPOB, sample prediction and query dataset from non-overlapping splits
                X_pred, y_pred, _, _ = sampler.sample(
                    batch_size=B,
                    num_total_points=config.train.num_prediction_points,
                    search_space_id=str(config.data.search_space_id),
                    standardize=config.data.standardize,
                    train_split="pred",
                    split="train",
                    device=config.experiment.device,
                )
                X_ac, y_ac, Xopt, yopt = sampler.sample(
                    num_total_points=config.train.num_query_points,
                    batch_size=B,
                    search_space_id=str(config.data.search_space_id),
                    standardize=config.data.standardize,
                    train_split="acq",
                    split="train",
                    device=config.experiment.device,
                )

            # reuse a query dataset for `num_trajectories` times
            X_ac = X_ac.tile(
                config.train.n_trajectories, 1, 1
            )  # (B_t := B * n_tra, num_query, d_x)
            y_ac = y_ac.tile(config.train.n_trajectories, 1, 1)
            Xopt = Xopt.tile(config.train.n_trajectories, 1, 1)
            yopt = yopt.tile(config.train.n_trajectories, 1, 1)

            # generate the candidate query pair set
            query_pair_idx, query_pair, query_pair_y, query_c = generate_pair_set(
                X=X_ac,
                y=y_ac,
                num_total_points=config.train.num_query_points,
                n_random_pairs=config.train.n_random_pairs,
                p_noise=config.train.p_noise,
                ranking_reward=config.train.ranking_reward,
            )

            # mask to register queried pairs
            mask = torch.ones(
                (query_pair.shape[0], query_pair.shape[1], 1)
            ).bool()  # (B_t, num_query_pairs, 1)

            # sample starting pairs from Q
            (context_pairs, context_pairs_y, context_c, init_pair_idx) = (
                sample_random_pairs_from_Q(
                    pair=query_pair,
                    pair_y=query_pair_y,
                    c=query_c,
                    num_samples=config.train.num_init_pairs,
                )
            )
            mask[:, init_pair_idx] = 0.0  # NOTE mask out sampled ones

            # policy learning setup
            rewards = []
            entropys = []
            log_probs = []
            rewards = []

        # generate prediction set
        _, src_pairs, src_pairs_y, src_c = generate_pair_set(
            X=X_pred,
            y=y_pred,
            num_total_points=config.train.num_prediction_points,
            n_random_pairs=config.train.num_prediction_points,
            p_noise=config.train.p_noise,
        )

        X_pred = X_pred.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        del X_pred
        del y_pred
        torch.cuda.empty_cache()

        cls_losses = 0.0

        # optimization loop with T time budget
        for t in range(1, config.train.max_T + 2):
            # prediction task: randomly split the prediction set into context and target
            _, ctx_idx, _, tar_idx = get_random_split_data(
                total_num=config.train.num_prediction_points,
                min_num_ctx=config.data.min_num_ctx,
                max_num_ctx=config.data.max_num_ctx,
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

            # compute and record accuracy and kt-cor
            acc = accuracy(f=pred_tar_f.detach(), c=src_c[:, tar_idx], reduce=True)
            kt_cor = kendalltau_correlation(
                pred_tar_f.detach(),
                src_pairs_y[:, tar_idx].view(len(src_pairs_y), -1, 1),
                reduce=True,
            )
            ravg.update("acc", acc)
            ravg.update("kt_cor", kt_cor)

            # optimization task
            if epoch > config.train.n_burnin:
                # take action (propose the next query) given current observations
                acq_values, next_pair_idx, log_prob, entropy = action(
                    model=model,
                    context_pairs=context_pairs,
                    context_preference=context_c,
                    t=t,
                    T=config.train.max_T,
                    X_pending=X_ac,  # all the query points
                    pair_idx=query_pair_idx.tile(
                        (n_gpus, 1)
                    ),  # NOTE DataParallel will split every tensor along the first dim
                    mask=mask,
                )
                # compute and record reward for previous run
                if t > 1:
                    reward = get_reward(
                        context_pairs_y=context_pairs_y,
                        acq_values=acq_values,
                        pair_y=query_pair_y,
                        regret_option=config.train.regret_option,
                    )
                    rewards.append(reward)
                    if t > config.train.max_T:
                        break

                # update observations and candidate set given the proposed query
                next_pair_idx = next_pair_idx[:, None, None]  # (B, 1, 1)

                # mask out the query pair from candidate query pair set
                mask.scatter_(
                    dim=1,
                    index=next_pair_idx,
                    src=torch.zeros_like(mask, device=mask.device),
                )

                # the query with corresponding preference label
                next_pair = gather_data_at_index(data=query_pair, idx=next_pair_idx)
                next_pair_y = gather_data_at_index(
                    data=query_pair_y, idx=next_pair_idx
                )  # NOTE only for computing the reward; won't be accessed by model
                next_c = gather_data_at_index(data=query_c, idx=next_pair_idx)

                # update observations
                context_pairs = torch.cat((context_pairs, next_pair), dim=1)
                context_c = torch.cat((context_c, next_c), dim=1)
                context_pairs_y = torch.cat((context_pairs_y, next_pair_y), dim=1)

                # record log probabilities and entropy
                log_probs.append(log_prob)
                entropys.append(entropy)

        # averaged BCE loss
        cls_loss = cls_losses / config.train.max_T

        # compute the policy learning loss given trajectories information
        if epoch > config.train.n_burnin:
            # the best observed value, max f_T
            best_y = torch.max(
                context_pairs_y.flatten(start_dim=1), dim=-1
            ).values.squeeze()
            opt_y = torch.max(query_pair_y.flatten(start_dim=1), dim=-1).values

            # record the final simple regret
            final_simple_regret = (opt_y - best_y).mean().item()

            # compute the policy learning loss
            policy_loss, entropy = finish_episode(
                entropys=entropys,
                rewards=rewards,
                log_probs=log_probs,
                discount_factor=config.train.discount_factor,
            )

            # auxiliary_ratio controls the ratio of training steps with auxiliary loss. Default 1.0.
            lw = config.train.loss_weight * (
                config.train.auxiliary_ratio > torch.rand((1), device=device)
            )
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

        # record metrics
        ravg.update("loss", loss)
        ravg.update("cls_loss", cls_loss)
        ravg.update("policy_loss", policy_loss)
        ravg.update("final_simple_regret", final_simple_regret)
        ravg.update("entropy", entropy)

        if epoch % config.train.print_freq == 0:
            line = f"{config.experiment.expid} step {epoch} "
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)
            ravg.reset()

        if config.experiment.wandb:
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
        if epoch % config.train.save_freq == 0 or epoch == config.train.n_steps:
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

            # # save to the experiment directory
            # torch.save(ckpt, "ckpt.tar")

            # save to the result saving path
            torch.save(ckpt, os.path.join(root, f"ckpt.tar"))

            # save to wandb if used
            if config.experiment.wandb:
                save_artifact(
                    run=wandb.run,
                    local_path=os.path.join(root, f"ckpt.tar"),
                    name="checkpoint",
                    type="model",
                )

        # downsize the batch and lower learning rate when introducing acquisition task
        if epoch == config.train.n_burnin:
            B = config.train.ac_train_batch_size
            for g in optimizer.param_groups:
                g["lr"] = config.train.ac_lr

        epoch += 1


if __name__ == "__main__":
    main()
