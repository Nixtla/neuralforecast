# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.timellm.ipynb.

# %% auto 0
__all__ = ['IS_TRANSFORMERS_INSTALLED', 'ReplicationPad1d', 'TokenEmbedding', 'PatchEmbedding', 'FlattenHead',
           'ReprogrammingLayer', 'Normalize', 'TimeLLM']

# %% ../../nbs/models.timellm.ipynb 6
import math
from typing import Optional

import torch
import torch.nn as nn

from ..common._base_windows import BaseWindows

from ..losses.pytorch import MAE

IS_TRANSFORMERS_INSTALLED = True
try:
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
except ImportError:
    IS_TRANSFORMERS_INSTALLED = False
    print("The transformers library is required for Time-LLM to work")

# %% ../../nbs/models.timellm.ipynb 9
class ReplicationPad1d(nn.Module):
    def __init__(self, padding):
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(
        self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1
    ):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Normalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

# %% ../../nbs/models.timellm.ipynb 11
class TimeLLM(BaseWindows):
    """TimeLLM

    Time-LLM is a reprogramming framework to repurpose an off-the-shelf LLM for time series forecasting.

    It trains a reprogramming layer that translates the observed series into a language task. This is fed to the LLM and an output
    projection layer translates the output back to numerical predictions.

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `input_size`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `patch_len`: int=16, length of patch.<br>
    `stride`: int=8, stride of patch.<br>
    `d_ff`: int=128, dimension of fcn.<br>
    `top_k`: int=5, top tokens to consider.<br>
    `d_llm`: int=768, hidden dimension of LLM.<br>
    `d_model`: int=32, dimension of model.<br>
    `n_heads`: int=8, number of heads in attention layer.<br>
    `enc_in`: int=7, encoder input size.<br>
    `dec_in`: int=7, decoder input size.<br>
    `llm` = None, LLM model to use. If not specified, it will use GPT-2 from https://huggingface.co/openai-community/gpt2"<br>
    `llm_config` = None, configuration of LLM. If not specified, it will use the configuration of GPT-2 from https://huggingface.co/openai-community/gpt2"<br>
    `llm_tokenizer` = None, tokenizer of LLM. If not specified, it will use the GPT-2 tokenizer from https://huggingface.co/openai-community/gpt2"<br>
    `llm_num_hidden_layers` = 32, hidden layers in LLM
    `llm_output_attention`: bool = True, whether to output attention in encoder.<br>
    `llm_output_hidden_states`: bool = True, whether to output hidden states.<br>
    `prompt_prefix`: str=None, prompt to inform the LLM about the dataset.<br>
    `dropout`: float=0.1, dropout rate.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    -[Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen. "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"](https://arxiv.org/abs/2310.01728)

    """

    SAMPLING_TYPE = "windows"

    def __init__(
        self,
        h,
        input_size,
        patch_len: int = 16,
        stride: int = 8,
        d_ff: int = 128,
        top_k: int = 5,
        d_llm: int = 768,
        d_model: int = 32,
        n_heads: int = 8,
        enc_in: int = 7,
        dec_in: int = 7,
        llm=None,
        llm_config=None,
        llm_tokenizer=None,
        llm_num_hidden_layers=32,
        llm_output_attention: bool = True,
        llm_output_hidden_states: bool = True,
        prompt_prefix: Optional[str] = None,
        dropout: float = 0.1,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        loss=MAE(),
        valid_loss=None,
        learning_rate: float = 1e-4,
        max_steps: int = 5,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        num_lr_decays: int = 0,
        early_stop_patience_steps: int = -1,
        scaler_type: str = "identity",
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        random_seed: int = 1,
        **trainer_kwargs,
    ):
        super(TimeLLM, self).__init__(
            h=h,
            input_size=input_size,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            futr_exog_list=futr_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            **trainer_kwargs,
        )

        # Asserts
        if stat_exog_list is not None:
            raise Exception("TimeLLM does not support static exogenous variables")
        if futr_exog_list is not None:
            raise Exception("TimeLLM does not support future exogenous variables")
        if hist_exog_list is not None:
            raise Exception("TimeLLM does not support historical exogenous variables")

        # Architecture
        self.patch_len = patch_len
        self.stride = stride
        self.d_ff = d_ff
        self.top_k = top_k
        self.d_llm = d_llm
        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = n_heads
        self.enc_in = enc_in
        self.dec_in = dec_in

        self.llm_config = llm_config
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer

        if self.llm is None:
            if not IS_TRANSFORMERS_INSTALLED:
                raise ImportError(
                    "Please install `transformers` to use the default LLM"
                )

            print(
                "Using GPT2 model as default and ignoring `llm_config` and `llm_tokenizer`"
            )

            self.llm_confg = GPT2Config.from_pretrained("openai-community/gpt2")
            self.llm = GPT2Model.from_pretrained(
                "openai-community/gpt2", config=self.llm_confg
            )
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        self.llm_num_hidden_layers = llm_num_hidden_layers
        self.llm_output_attention = llm_output_attention
        self.llm_output_hidden_states = llm_output_hidden_states
        self.prompt_prefix = prompt_prefix

        if self.llm_tokenizer.eos_token:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.llm_tokenizer.add_special_tokens({"pad_token": pad_token})
            self.llm_tokenizer.pad_token = pad_token

        for param in self.llm.parameters():
            param.requires_grad = False

        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.dropout
        )

        self.word_embeddings = self.llm.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1024
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            self.d_model, self.n_heads, self.d_ff, self.d_llm
        )

        self.patch_nums = int((input_size - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            self.enc_in, self.head_nf, self.h, head_dropout=self.dropout
        )

        self.normalize_layers = Normalize(self.enc_in, affine=False)

    def forecast(self, x_enc):

        x_enc = self.normalize_layers(x_enc, "norm")

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>{self.prompt_prefix}"
                f"Task description: forecast the next {str(self.h)} steps given the previous {str(self.input_size)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.llm_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.llm.get_input_embeddings()(
            prompt.to(x_enc.device)
        )  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32))
        enc_out = self.reprogramming_layer(
            enc_out, source_embeddings, source_embeddings
        )
        llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm(inputs_embeds=llm_enc_out).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        )
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums :])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]

        x = insample_y.unsqueeze(-1)

        y_pred = self.forecast(x)
        y_pred = y_pred[:, -self.h :, :]
        y_pred = self.loss.domain_map(y_pred)

        return y_pred
