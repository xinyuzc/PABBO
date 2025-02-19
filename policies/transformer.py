import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from torch.distributions import Normal
from typing import Optional, Tuple, Union


def build_mlp(dim_in: int, dim_hid: int, dim_out: int, depth: int):
    """Build MLP.
    Args:
        dim_in: input dimension.
        dim_hidden: hidden layer's dimension.
        dim_out: output dimension.
        depth: depth of MLP.
    """
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth - 2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class MyTransformerEncoderLayer(TransformerEncoderLayer):
    """Customized Transformer encoder layer to save computation,
    where self-attention between the full sequence and context sequence is the same as cross-attention.
    """

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:

        slice_ = attn_mask[0, :]
        zero_mask = slice_ == 0
        num_ctx = torch.sum(zero_mask).item()

        x = self.self_attn(
            x,
            x[:, :num_ctx, :],
            x[:, :num_ctx, :],
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)


class TransformerBase(nn.Module):
    def __init__(
        self,
        d_x: int,
        n_comp: int,
        d_model: int,
        nhead: int,
        dropout: float,
        n_layers: int,
        dim_feedforward: int,
        emb_depth: int,
        tok_emb_option: str = "ind_point_emb_sum",
        transformer_encoder_layer_cls="efficient",
    ):
        """Transformer base.
        Args:
            d_x, scalar: input dimension.
            n_comp, scalar: number of alternatives in a query, fixed as 2.
            d_model, scalar: Transformer input dimension.
            nhead, dropout, n_layers, dim_feedforward: architecture hyperparameters for Transformer encoder.
            emb_depth, scalar: number of hidden layers in embedder.
            token_emb_option, str: the method for embedding the tokens. Default to be ``ind_point_emb_sum`` which is presented in the paper
            transformer_encoder_layer_cls, str: the class of Transformer encoder layer.

        Attrs:
            d_c, scalar: preference label dimenion, fixed as 1.
        """
        super().__init__()
        self.d_x = d_x
        self.n_comp = n_comp
        self.d_c = 1
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.emb_depth = emb_depth
        self.tok_emb_option = tok_emb_option
        self.setup_tok_embedder()
        self.transformer_encoder_layer = (
            MyTransformerEncoderLayer
            if transformer_encoder_layer_cls == "efficient"
            else nn.TransformerEncoderLayer
        )
        encoder_layer = self.transformer_encoder_layer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def setup_tok_embedder(self):
        # set up token embedder
        if self.tok_emb_option == "ind_point_emb_sum":
            self.x_embedders = nn.ModuleList(
                [
                    build_mlp(self.d_x, self.d_model, self.d_model, self.emb_depth)
                    for _ in range(self.n_comp)
                ]
            )  # embed instances in a query with different embedders
            self.c_embedder = nn.Embedding(self.n_comp + 1, self.d_model)  #
        else:
            raise ValueError(f"Remove for simplicity.")

    def embed_input(
        self,
        query_src: torch.Tensor,
        c_src: torch.Tensor,
        eval_pos: torch.Tensor,
        num_src: int,
        num_eval: int,
    ) -> torch.Tensor:
        """Input embedding.
        Args: let `B` be the batch size, `num_src` be the number of context pairs, `num_eval` be the number of target locations,
            query_src, [B, num_src, n_comp * d_x]: a batch of context queries, each consists of `num_src` queries with `n_comp` alternatives.
            c_src, [B, num_src, d_c]: preference label associated with context pairs.
            eval_pos, [B, num_eval, d_x]: a batch of target locations, each consists of `num_eval` locations.
            num_eval, scalar: the number of target locations. (dummy)
        Returns:
            emb, [B, num_src + num_eval, d_model]: embeddings for context pairs and target locations.
        """
        if self.tok_emb_option == "ind_point_emb_sum":
            # Given context queries with their preference label \[(x_{i, 1}, x_{i, 2}, l_i)\], and a target location \[x_i\],
            emb = self.x_embedders[0](eval_pos)  # For target locations x_i, E = E_{x_i}
            if num_src > 0:
                c_src = c_src.squeeze(-1)
                c_src_emb = self.c_embedder(
                    c_src.int()
                )  # preference label embedding, E_{l_i}
                query_src_embs = [
                    self.x_embedders[i](
                        query_src[:, :, i * self.d_x : (i + 1) * self.d_x]
                    )
                    for i in range(self.n_comp)
                ]  # context location embedding, E_{x_{i, 1}}, E_{x_{i, 2}}
                src_emb = torch.sum(
                    torch.stack([*query_src_embs, c_src_emb], dim=0), dim=0
                )  # For context duel, E = E_{x_{i, 1}} \oplus E_{x_{i, 2}} \oplus E_{l_i}
                emb = torch.concat(
                    (src_emb, emb), dim=1
                )  # concatenate embeddings for context and target as the returns
        else:
            raise ValueError("Remove for simplicity.")
        return emb

    def create_mask(self, query_src, eval_pos, device) -> Tuple[torch.Tensor, int, int]:
        """create mask for tokens, where context tokens can be attended to by all tokens, while each target token can only be attended by itself.
        Args:
            query_src, [B, num_src, n_comp * d_x]: a batch of context queries, each consists of `num_src` queries with `n_comp` alternatives.
            eval_pos, [B, num_eval, d_x]: a batch of target locations, each consists of `num_eval` locations.
        Returns:
            mask, [B, num_src + num_eval, num_src + num_eval]: mask for the context and target sequence.
            num_src, scalar: number of context queries.
            num_eval, scalar: number of target locations.
        """
        num_src = query_src.shape[1]
        num_eval = eval_pos.shape[1]
        num_all = num_src + num_eval

        mask = torch.zeros((num_all, num_all), device=device).fill_(float("-inf"))
        mask[:, :num_src] = 0.0
        torch.diagonal(mask[num_src:num_all, num_src:num_all], 0).zero_()
        return mask, num_src, num_eval

    def encode(
        self, query_src: torch.Tensor, c_src: torch.Tensor, eval_pos: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Encode data with Transformer blocks.
        Args:
            query_src, [B, num_src, n_comp * d_x]: a batch of context queries, each consists of `num_src` queries with `n_comp` alternatives.
            c_src, [B, num_src, d_c]: preference label associated with context pairs.
            eval_pos, [B, num_eval, d_x]: a batch of target locations, each consists of `num_eval` locations.

        Returns:
            out, [B, num_src + num_eval, d_model): Transformer outputs.
            num_eval, scalar: number of target locations.
            emb, [B, num_src + num_eval, d_model]: the input embedding.
        """
        device = query_src.device
        mask, num_src, num_eval = self.create_mask(query_src, eval_pos, device=device)
        emb = self.embed_input(query_src, c_src, eval_pos, num_src, num_eval)
        out = self.encoder(emb, mask=mask)
        return out, num_eval, emb


class TransformerModel(TransformerBase):
    def __init__(
        self,
        d_x: int,
        d_f: int,
        d_model: int,
        nhead: int,
        dropout: float,
        n_layers: int,
        dim_feedforward: int,
        emb_depth: int,
        tok_emb_option: str = "ind_point_emb_sum",
        transformer_encoder_layer_cls="efficient",
        joint_model_af_training: bool = True,
        use_value_network: bool = False,
        af_name: str = "mlp",
        bound_std: bool = False,
        nbuckets: int = 2,
        y_range: float = 3.0,
        time_budget: bool = True,
    ):
        """Preferential amortized black-box optimization (PABBO).

        Args:
            d_x, scalar: input dimension.
            d_f, scalar: function output dimension, fixed as 1.
            d_model, scalar: Transformer input dimension.
            nhead, dropout, n_layers, dim_feedforward: architecture hyperparameters for Transformer encoder.
            emb_depth, scalar: number of hidden layers in embedder.
            token_emb_option, str: the method for embedding the tokens. Default to be ``ind_point_emb_sum`` which is presented in the paper
            transformer_encoder_layer_cls, str: the class of Transformer encoder layer.
            joint_model_af_training, bool: whether to jointly train transformer. Default as True.
            use_value_network, bool: whether to use value network (dummy)
            af_name, str: acquisition function. Default as "mlp".
            bound_std, bool: whether to bound the standard deviation range of the Gaussian prediction head. Default as False.
            nbuckets, scalar: prediction head output dimension. Default as 2 and a Gaussian prediction head is used.
            y_range, float: to obtain predicted function values' range,(-y_range, y_range), when setting up Riemann distribution (dummy).
            time_budget, bool: whether to pass t / T to the acquisition head.
        """
        super().__init__(
            d_x=d_x,
            n_comp=2,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            emb_depth=emb_depth,
            tok_emb_option=tok_emb_option,
            transformer_encoder_layer_cls=transformer_encoder_layer_cls,
        )
        self.d_f = d_f
        self.bound_std = bound_std
        self.af_name = af_name
        self.joint_model_af_training = joint_model_af_training
        self.use_value_network = use_value_network
        self.nbuckets = nbuckets
        self.y_range = y_range
        self.time_budget = time_budget

        if self.nbuckets > 2:
            raise ValueError("Remove for simplicity.")

        self.eval_bucket_decoder = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.nbuckets),
        )  # prediction head

        # acquisition head
        if self.af_name == "mlp":
            if time_budget:
                self.af_mlp = build_mlp(2 * self.d_model + 1, dim_feedforward, 1, 3)
            else:
                self.af_mlp = build_mlp(2 * self.d_model, dim_feedforward, 1, 3)

    def forward(
        self,
        query_src: torch.Tensor,
        c_src: torch.Tensor,
        eval_pos: torch.Tensor,
        acquire: bool = True,
        acquire_comb_idx: Union[torch.Tensor, None] = None,
        t: int = 1,
        T: int = 1,
        best_y: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform prediction / acquisition.
        Args:
            query_src, [B, num_src, n_comp * d_x]: a batch of context queries, each consists of `num_src` queries with `n_comp` alternatives.
            c_src, [B, num_src, d_c]: preference label associated with context pairs.
            eval_pos, [B, num_eval, d_x]: a batch of target locations, each consists of `num_eval` locations.
            acquire, bool: whether to conduct acquisition task or prediction task on `eval_pos`.
            acquire_comb_idx, [1, num_query_pairs, 2]: indices from the full combinations of target locations, so as to create query pair set Q.
            t, scalar: current optimization step.
            T, scalar: horizon budget.
            best_y [B, ]: (dummy)

        Returns:
            eval_af [B, num_query_pairs, 1]: acquistion values for all pairs in the query pair set.
            eval_f [B, num_eval, 1]: Gaussian mean for function distribution.
            eval_f_std [B, num_eval, 1]: Gaussian variance for function distribution.
            eval_vf [B, ]: (dummy)
        """
        B, num_eval = eval_pos.shape[:-1]
        device = query_src.device

        eval_af, eval_vf = None, None
        eval_f, eval_f_std = None, None

        z, num_eval, _ = self.encode(
            query_src=query_src, c_src=c_src, eval_pos=eval_pos
        )  # encode input sequence
        _, z_eval = (
            z[:, :-num_eval],
            z[:, -num_eval:],
        )  # we only need the Transformer output for target locations

        if acquire:  # acquisition task
            if acquire_comb_idx is None:  # use the full combinations
                acquire_comb_idx = torch.combinations(
                    torch.arange(num_eval, device=device)
                )
            else:  # take a subset
                acquire_comb_idx = acquire_comb_idx.view(-1, 2)

            z_eval = z_eval[:, acquire_comb_idx]  # pair all the Transformer outputs

            state_ix = (t / T) * torch.ones(
                (B, 1), device=device
            )  # state of optimization step
            if (
                self.af_name == "mlp"
            ):  # state: [concatenated Trasformer_outputs of the pair, t/T]
                mlp_in = z_eval.flatten(start_dim=-2)
                if self.time_budget:  # pass t/T to the acquisition head
                    state_ix_af = state_ix[:, None, :].tile(1, len(acquire_comb_idx), 1)
                    mlp_in = torch.cat([mlp_in, state_ix_af], dim=-1)

                if not self.joint_model_af_training:
                    mlp_in = mlp_in.detach()

                eval_af = self.af_mlp(mlp_in)
            else:
                raise ValueError(f"Remove for simplicity.")

            if self.use_value_network:
                eval_vf = self.value_head(state_ix).squeeze()

        else:  # prediction task
            eval_bucket_output = self.eval_bucket_decoder(z_eval)

            if self.nbuckets == 1:  # point estimate
                eval_f = eval_bucket_output
                eval_f_std = torch.zeros_like(eval_f, device=device)
            elif self.nbuckets == 2:  # Gaussian head
                eval_f_std = F.softplus(eval_bucket_output[:, :, [1]])
                dist = Normal(loc=eval_bucket_output[:, :, [0]], scale=eval_f_std)
                eval_f = dist.rsample()
            else:
                raise ValueError("Remove for simplicity.")
        return eval_af, eval_f, eval_f_std, eval_vf
