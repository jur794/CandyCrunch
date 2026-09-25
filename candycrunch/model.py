
import numpy as np
import random
from torch import flatten
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
import copy
import inspect
# print(torch.__version__)
# print(inspect.getsource(nn.TransformerEncoderLayer))

def remove_low_intensity_peaks(array, removal_threshold, removal_percentage):
  candidate_indices = np.where(np.logical_and(array > 0.0001, array <= removal_threshold))[0]
  indices_to_remove = np.random.choice(candidate_indices, round(removal_percentage*len(candidate_indices)))
  array_copy = np.copy(array)
  array_copy[indices_to_remove] = 0
  return array_copy


def peak_intensity_jitter(array, augment_intensity):
  return array * np.random.uniform(1 - augment_intensity, 1 + augment_intensity, len(array)).astype(np.float32)


def new_peak_addition(array, n_noise_peaks, max_noise_intensity):
  idx_noise_peaks = np.random.choice(np.where(array == 0)[0], n_noise_peaks)
  new_values = max_noise_intensity * np.random.random(len(idx_noise_peaks))
  noisy_array = np.copy(array)
  noisy_array[idx_noise_peaks] = new_values
  return noisy_array


transform_mz = transforms.Compose([
  lambda x: remove_low_intensity_peaks(x, removal_threshold = 0.008, removal_percentage = 0.1),
  lambda x: peak_intensity_jitter(x, augment_intensity = 0.25),
  lambda x: new_peak_addition(x, n_noise_peaks = 10, max_noise_intensity = 0.005)
])


def rt_jitter(RT):
  return max(0, RT + random.uniform(-0.1, 0.1))


transform_rt = transforms.Compose([
    lambda x: rt_jitter(x)
])



class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform_mz=None, transform_rt=None):
        self.x = x
        self.y = y
        self.transform_mz = transform_mz
        self.transform_rt = transform_rt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        mz = self.x[index][0]
        peak_list = self.x[index][1]
        mz_r = self.x[index][2]
        prec = self.x[index][3]
        glycan_type = self.x[index][4]
        RT = self.x[index][5]
        mode = self.x[index][6]
        lc = self.x[index][7]
        modification = self.x[index][8]
        trap = self.x[index][9]
        out = self.y[index]

        if self.transform_mz:
            mz = self.transform_mz(mz)

        if self.transform_rt:
            RT = self.transform_rt(RT)

        return (
            torch.FloatTensor(mz),
            torch.FloatTensor(peak_list),
            torch.FloatTensor(mz_r),
            torch.FloatTensor(prec),
            torch.LongTensor([glycan_type]),
            torch.FloatTensor([RT]),
            torch.LongTensor([mode]),
            torch.LongTensor([lc]),
            torch.LongTensor([modification]),
            torch.LongTensor([trap]),
            torch.LongTensor([out]),
        )

class SeqSimpleDataset(SimpleDataset):
    """SimpleDataset (CNN-style features + augmentation) plus BPE-tokenized decoder
    targets, for training CandyCrunch_CNN_Decoder. Keeps the integer class label
    (y_class) around too, for the optional auxiliary classification loss.

    y_class: list[int]  -- glycans.index(...) labels, same as SimpleDataset's `y`
    y_str:   list[str]  -- raw IUPAC strings, tokenized once here at construction
    """

    def __init__(self, x, y_class, y_str, tokenizer, transform_mz=None, transform_rt=None,
                 max_target_len=None):
        super().__init__(x, y_class, transform_mz=transform_mz, transform_rt=transform_rt)
        self.pad_id = tokenizer.vocab[tokenizer.special_tokens['pad_token']]
        ids = tokenizer.encode(list(y_str))["token_ids"]
        # BPETokenizer.encode() pads to its fixed max_seq_length (512); trim down to the
        # longest real sequence in this set so the decoder isn't wasting compute on padding.
        nonpad = (ids != self.pad_id).any(dim=0).nonzero().max().item() + 1
        if max_target_len is not None:
            nonpad = min(nonpad, max_target_len)
        self.encoded_targets = ids[:, :nonpad].contiguous()
        self.target_len = nonpad

    def __getitem__(self, index):
        base = super().__getitem__(index)
        tgt = self.encoded_targets[index]
        decoder_input_ids = tgt[:-1].clone()
        target_labels = tgt[1:].clone()
        tgt_key_padding_mask = (decoder_input_ids == self.pad_id)
        return base + (decoder_input_ids, tgt_key_padding_mask, target_labels)


class TransDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform_rt=None):
        self.x = x
        self.y = y
        self.transform_rt = transform_rt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        peak_list = self.x[index][1]
        prec = self.x[index][3]
        glycan_type = self.x[index][4]
        RT = self.x[index][5]
        mode = self.x[index][6]
        lc = self.x[index][7]
        modification = self.x[index][8]
        trap = self.x[index][9]
        out = self.y[index]

        if self.transform_rt:
            RT = self.transform_rt(RT)

        peak_list = torch.FloatTensor(peak_list)

        # True means "ignore this token" for PyTorch Transformer padding masks.
        peak_padding_mask = peak_list.abs().sum(dim=-1) == 0

        return (
            peak_list,
            peak_padding_mask,
            torch.FloatTensor(prec),
            torch.LongTensor([glycan_type]),
            torch.FloatTensor([RT]),
            torch.LongTensor([mode]),
            torch.LongTensor([lc]),
            torch.LongTensor([modification]),
            torch.LongTensor([trap]),
            torch.LongTensor([out]),
        )


class PeakResBlock(nn.Module):
    def __init__(self, peak_hidden_dim, dropout=0.2, kernel_size=3, dilations=(1, 2, 4, 8), causal=False):
        super().__init__()
        self.resunits = nn.Sequential(*[ResUnit(in_channels=peak_hidden_dim, size=kernel_size, dilation=dilation,
                                                causal=causal, in_ln=True,) for dilation in dilations])
        self.dropout = nn.Dropout(dropout)
    def forward(self, peak_features, peak_padding_mask=None):
        if peak_padding_mask is not None:
            peak_features = peak_features.masked_fill( peak_padding_mask.unsqueeze(-1), 0.0)
        peak_features = peak_features.transpose(1, 2)
        peak_features = self.resunits(peak_features)
        peak_features = peak_features.transpose(1, 2)
        peak_features = self.dropout(peak_features)
        if peak_padding_mask is not None:
            peak_features = peak_features.masked_fill(peak_padding_mask.unsqueeze(-1),0.0)
        return peak_features

class ResUnit(nn.Module):
    """Daniel new Change"""
    def __init__(self, in_channels, size=3, dilation=1, causal=False, in_ln=True,
                 se=False, se_reduction=8):
        super(ResUnit, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.in_ln = in_ln
        # 1. InstanceNorm1d
        self.se = se
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine=True)
            self.ln1.weight.data.fill_(1.0)
        # 2. Bottleneck 1×1 convolution, Reduces channels: C -> C/2
        self.conv_in = nn.Conv1d(in_channels, in_channels // 2, 1)
        # 3. InstanceNorm1d
        self.ln2 = nn.InstanceNorm1d(in_channels // 2, affine=True)
        self.ln2.weight.data.fill_(1.0)
        # 4. Dilated Conv1D
        # 5. Optional causal convolution, the padding part
        padding = dilation * (size - 1) if causal else dilation * (size - 1) // 2
        self.conv_dilated = nn.Conv1d( in_channels // 2, in_channels // 2, size, dilation=dilation, padding=padding)
        # 6. InstanceNorm1d
        self.ln3 = nn.InstanceNorm1d(in_channels // 2, affine=True)
        self.ln3.weight.data.fill_(1.0)
        # 7. Bottleneck 1×1 convolution, Restores channels: C/2 -> C
        self.conv_out = nn.Conv1d(in_channels // 2, in_channels, 1)
        if self.se:
            se_hidden = max(in_channels // se_reduction, 1)
            self.se_fc1 = nn.Conv1d(in_channels, se_hidden, 1)
            self.se_fc2 = nn.Conv1d(se_hidden, in_channels, 1)
    def forward(self, inp):
        x = inp
        if self.in_ln:
            x = self.ln1(x)
        x = nn.functional.leaky_relu(x)
        x = nn.functional.leaky_relu(self.ln2(self.conv_in(x)))
        x = self.conv_dilated(x)
        if self.causal and self.size > 1:
            x = x[:, :, :-self.dilation * (self.size - 1)]
        x = nn.functional.leaky_relu(self.ln3(x))
        x = self.conv_out(x)
        if self.se:
            s = x.mean(dim=2, keepdim=True) # (B, 64, L) -> (B, 64, 1)
            s = nn.functional.leaky_relu(self.se_fc1(s)) # (B, 64, 1) -> (B, 8, 1)
            s = torch.sigmoid(self.se_fc2(s)) # (B, 8, 1) -> (B, 64, 1)
            x = x * s # (B, 64, L) -> (B, 64, L)
        # 8. Residual connection
        out = x + inp
        return out


class CandyCrunch_CNN(nn.Module):
    """Daniel new Change"""
    def __init__( self, input_dim, num_classes=1, hidden_dim=512, input_precursor_dim=None, dropout=0.2,
                  se=True, se_reduction=8):
        super(CandyCrunch_CNN, self).__init__()

        self.input_dim = input_dim
        self.type_emb = nn.Embedding(5, 24)
        self.mode_emb = nn.Embedding(3, 24)
        self.lc_emb = nn.Embedding(4, 24)
        self.modification_emb = nn.Embedding(4, 24)
        self.trap_emb = nn.Embedding(5, 24)
        self.prec_block = nn.Sequential( nn.Linear(input_precursor_dim, 24), nn.LayerNorm(24), nn.LeakyReLU())
        self.rt_block = nn.Sequential( nn.Linear(1, 24), nn.LayerNorm(24), nn.LeakyReLU())

        self.res_block = nn.Sequential( nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1),
                                        nn.LeakyReLU(),
                                        ResUnit(64, size=3, dilation=1, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=2, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=4, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=8, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=16, causal=False, se=se, se_reduction=se_reduction),
                                        ResUnit(64, size=3, dilation=32, causal=False, se=se, se_reduction=se_reduction),
                                        nn.AdaptiveMaxPool1d(102))
        self.fc_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=6528, out_features=1024)
        self.comb_block1 = nn.Sequential(nn.Linear( 2 * hidden_dim + 24 + 24 + 24 + 24 + 24 + 24 + 24, 2 * 512,),
                                         nn.LayerNorm(2 * 512),
                                         nn.LeakyReLU(),
                                         nn.Dropout(dropout))

        self.comb_lin1 = nn.Linear(2 * 512, 2 * 256)
        self.comb_block2 = nn.Sequential(nn.LayerNorm(2 * 256),
                                         nn.LeakyReLU(),
                                         nn.Dropout(dropout))
        self.comb_lin2 = nn.Linear(2 * 256, num_classes)

    def forward(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap, rep=False,
                return_feature_map=False):
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_block(precursor)
        rt = self.rt_block(rt)
        mz = self.res_block(mz_features)
        # Per-position conv feature map (before pooling to a single vector), for callers that
        # want to cross-attend to it (e.g. CandyCrunch_CNN_Decoder's "feature_map" memory mode).
        feature_map = mz.transpose(1, 2) if return_feature_map else None
        mz = flatten(mz, start_dim=1)
        mz = self.fc_dropout(mz)
        mz = F.leaky_relu(self.fc1(mz))
        comb = torch.cat([mz,precursor,glycan_type,rt,mode,lc,modification,trap],dim=1)
        comb = self.comb_block1(comb)
        comb_rep = self.comb_lin1(comb)
        comb = self.comb_block2(comb_rep)
        comb = self.comb_lin2(comb)
        if return_feature_map:
            return comb, comb_rep, feature_map
        if rep:
            return comb, comb_rep
        else:
            return comb


class CandyCrunch_CNN_Decoder(nn.Module):
    """Autoregressive IUPAC-string decoder cross-attending to a pretrained CandyCrunch_CNN's
    spectrum representation.

    memory_mode="simple": memory is the single penultimate vector (`comb_rep`, 512-dim,
    already exposed by CandyCrunch_CNN via `rep=True`) projected to one memory token.
    memory_mode="feature_map": memory is the pre-pool per-position conv feature map (102
    positions, exposed via `return_feature_map=True`) plus one metadata token, so the decoder
    can cross-attend to different regions of the spectrum. Swapping between the two only
    changes `encode()`'s memory construction -- the decoder, loss, and beam search are
    identical either way.
    """

    def __init__(self, base_model_kwargs, vocab_size, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=1024, dropout=0.1, memory_mode="simple", max_target_len=512,
                 activation="gelu", pad_token_id=0, bos_token_id=1, eos_token_id=2):
        super().__init__()
        if memory_mode not in ("simple", "feature_map"):
            raise ValueError(f"Unknown memory_mode={memory_mode!r}. Use 'simple' or 'feature_map'.")

        self.cnn = CandyCrunch_CNN(**base_model_kwargs)
        self.memory_mode = memory_mode
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        rep_dim = self.cnn.comb_lin1.out_features           # 512
        metadata_dim = self.cnn.type_emb.embedding_dim       # 24
        fmap_channels = self.cnn.res_block[-2].conv_out.out_channels   # 64, last ResUnit before the pool

        if memory_mode == "simple":
            self.rep_proj = nn.Linear(rep_dim, d_model)
        else:
            self.fmap_proj = nn.Linear(fmap_channels, d_model)
            self.fmap_pos_emb = nn.Embedding(102, d_model)
            self.meta_proj = nn.Linear(7 * metadata_dim, d_model)

        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_pos_emb = nn.Embedding(max_target_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, activation=activation, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))
        self.fc_out = nn.Linear(d_model, vocab_size)

    def load_pretrained_cnn(self, checkpoint_path, strict=True):
        """Loads CandyCrunch_CNN weights from an existing CNN checkpoint into self.cnn.

        Returns the checkpoint dict so the caller can assert compatibility (glycan label set,
        comp_vector_order, and -- critically -- that this checkpoint's train/test split matches
        the split being used to train the decoder; otherwise "validation" spectra here may have
        been training data for this encoder, silently inflating decoder validation accuracy).

        No freeze/unfreeze here by design: the decoder is currently trained end-to-end with the
        loaded encoder, not staged frozen-then-unfrozen. A frozen-encoder + differential-LR
        fine-tuning experiment is a deferred future option, not implemented now.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                     for k, v in state_dict.items()}
        self.cnn.load_state_dict(state_dict, strict=strict)
        return checkpoint

    def encode(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap):
        """Returns (memory, memory_key_padding_mask, class_logits).
        memory: (B, 1, d_model) in "simple" mode, (B, 103, d_model) in "feature_map" mode.
        memory_key_padding_mask: None -- every memory token is always valid.
        class_logits: (B, num_classes), the CNN's own classification head output (auxiliary loss).
        """
        need_fmap = (self.memory_mode == "feature_map")
        out = self.cnn(mz_features, precursor, glycan_type, rt, mode, lc, modification, trap,
                      rep=True, return_feature_map=need_fmap)
        class_logits, comb_rep = out[0], out[1]

        if not need_fmap:
            memory = self.rep_proj(comb_rep).unsqueeze(1)
        else:
            feature_map = out[2]                                          # (B, 102, 64)
            tokens = self.fmap_proj(feature_map)                          # (B, 102, d)
            pos = torch.arange(tokens.size(1), device=tokens.device)
            tokens = tokens + self.fmap_pos_emb(pos).unsqueeze(0)
            meta_cat = torch.cat([
                self.cnn.prec_block(precursor),
                self.cnn.type_emb(glycan_type).squeeze(1),
                self.cnn.rt_block(rt),
                self.cnn.mode_emb(mode).squeeze(1),
                self.cnn.lc_emb(lc).squeeze(1),
                self.cnn.modification_emb(modification).squeeze(1),
                self.cnn.trap_emb(trap).squeeze(1),
            ], dim=1)                                                     # (B, 168)
            meta_token = self.meta_proj(meta_cat).unsqueeze(1)             # (B, 1, d)
            memory = torch.cat([tokens, meta_token], dim=1)                # (B, 103, d)

        return memory, None, class_logits

    def decode_step(self, memory, tgt_input_ids, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """memory: (B, S, d_model); tgt_input_ids: (B, T) Long -> logits (B, T, vocab_size)."""
        batch_size, seq_len = tgt_input_ids.shape
        positions = torch.arange(seq_len, device=tgt_input_ids.device)
        tgt = self.tgt_tok_emb(tgt_input_ids) + self.tgt_pos_emb(positions).unsqueeze(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tgt_input_ids.device, dtype=torch.bool)
        hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(hidden)

    def forward(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap,
                tgt_input_ids=None, tgt_key_padding_mask=None):
        """Returns (token_logits, class_logits). token_logits is None if tgt_input_ids is None
        (encode-only path)."""
        memory, memory_key_padding_mask, class_logits = self.encode(
            mz_features, precursor, glycan_type, rt, mode, lc, modification, trap)
        if tgt_input_ids is None:
            return None, class_logits
        token_logits = self.decode_step(memory, tgt_input_ids, tgt_key_padding_mask, memory_key_padding_mask)
        return token_logits, class_logits


#########################################################################################################
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def make_norm(norm_type, dim):
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    if norm_type == "rms":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type={norm_type!r}. Use 'layer' or 'rms'.")


def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation={name!r}.")


class DenseFeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.2, activation="gelu"):
        super().__init__()
        self.net = nn.Sequential( nn.Linear(d_model, ff_dim),
                                  make_activation(activation),
                                  nn.Dropout(dropout),
                                  nn.Linear(ff_dim, d_model))
    def forward(self, x, padding_mask=None):
        return self.net(x)
    def get_aux_loss(self):
        return None


class MoEFeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, num_experts=4, top_k=2, dropout=0.2, activation="gelu"):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.last_aux_loss = None
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, ff_dim),
                                                    make_activation(activation),
                                                    nn.Dropout(dropout),
                                                    nn.Linear(ff_dim, d_model)) for _ in range(num_experts)])

    def load_balancing_loss(self, gate_probs, top_indices, padding_mask=None):
        selected_experts = F.one_hot(top_indices, num_classes=self.num_experts).float().sum(dim=-2)
        if padding_mask is not None:
            valid_mask = (~padding_mask).float().unsqueeze(-1)
            valid_count = valid_mask.sum().clamp_min(1.0)
            importance = (gate_probs * valid_mask).sum(dim=(0, 1)) / valid_count
            load = (selected_experts * valid_mask).sum(dim=(0, 1)) / (valid_count * self.top_k)
        else:
            importance = gate_probs.mean(dim=(0, 1))
            load = selected_experts.mean(dim=(0, 1)) / self.top_k
        return self.num_experts * torch.sum(importance * load)

    def forward(self, x, padding_mask=None):
        gate_logits = self.router(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        self.last_aux_loss = self.load_balancing_loss(gate_probs=gate_probs,
                                                      top_indices=top_indices,
                                                      padding_mask=padding_mask)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        gather_index = top_indices.unsqueeze(-1).expand(*top_indices.shape, x.size(-1))
        selected_outputs = torch.gather(expert_outputs, dim=2, index=gather_index)
        return (selected_outputs * top_weights.unsqueeze(-1)).sum(dim=2)

    def get_aux_loss(self):
        return self.last_aux_loss


class CandyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model,nhead, ff_dim, encoder_type="dense", norm_type="layer", dropout=0.2,
                 activation="gelu", num_experts=4, moe_top_k=2, norm_first=True):
        super().__init__()

        self.norm_first = norm_first
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)

        self.attn = nn.MultiheadAttention( embed_dim=d_model,
                                           num_heads=nhead,
                                           dropout=dropout,
                                           batch_first=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if encoder_type == "dense":
            self.ff = DenseFeedForward(d_model=d_model,
                                       ff_dim=ff_dim,
                                       dropout=dropout,
                                       activation=activation)
        elif encoder_type == "moe":
            self.ff = MoEFeedForward(d_model=d_model,
                                     ff_dim=ff_dim,
                                     num_experts=num_experts,
                                     top_k=moe_top_k,
                                     dropout=dropout,
                                     activation=activation)
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}. Use 'dense' or 'moe'.")

    def _mask_padding(self, x, padding_mask):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if self.norm_first:
            attn_input = self.norm1(src)
            attn_out, _ = self.attn( attn_input, attn_input, attn_input, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal )
            src = src + self.dropout1(attn_out)
            src = self._mask_padding(src, src_key_padding_mask)

            ff_out = self.ff(self.norm2(src), padding_mask=src_key_padding_mask)
            src = src + self.dropout2(ff_out)
            src = self._mask_padding(src, src_key_padding_mask)
        else:
            attn_out, _ = self.attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                    need_weights=False, is_causal=is_causal)
            src = self.norm1(src + self.dropout1(attn_out))
            src = self._mask_padding(src, src_key_padding_mask)

            ff_out = self.ff(src, padding_mask=src_key_padding_mask)
            src = self.norm2(src + self.dropout2(ff_out))
            src = self._mask_padding(src, src_key_padding_mask)

        return src

    def get_aux_loss(self):
        return self.ff.get_aux_loss()


class CandyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.last_aux_loss = None

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        aux_losses = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
            aux_loss = layer.get_aux_loss()
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        if self.norm is not None:
            output = self.norm(output)
        if aux_losses:
            self.last_aux_loss = torch.stack(aux_losses).mean()
        else:
            self.last_aux_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        return output
    def get_aux_loss(self):
        return self.last_aux_loss


class CandyCrunch_Transformer(nn.Module):
    def __init__(self, num_classes=None, input_precursor_dim=None, heads=None, layers=None, ff_dim=None, peak_dim=2,
                 metadata_dim=24, dropout=0.2, peak_encoder="linear", encoder_type="dense",
                 norm_type="layer", activation="leaky_relu", encoder_activation="gelu", peak_hidden_dim=None,
                 lambda_min=10 ** -2.5, lambda_max=10 ** 3.3, use_resunits=False, res_kernel_size=3,
                 res_dilations=(1, 2, 4, 8), res_causal=False, num_experts=4, moe_top_k=2,
                 norm_first=True, encoder_final_norm=True):
        super().__init__()

        if ff_dim is None or heads is None or layers is None:
            raise ValueError("ff_dim, heads, and layers must be provided.")

        if peak_hidden_dim is None:
            peak_hidden_dim = ff_dim // 2

        if peak_hidden_dim % heads != 0:
            raise ValueError(
                f"peak_hidden_dim must be divisible by heads. "
                f"Got peak_hidden_dim={peak_hidden_dim}, heads={heads}.")

        self.peak_encoder = peak_encoder
        self.encoder_type = encoder_type
        self.norm_type = norm_type
        self.use_resunits = use_resunits
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        if peak_encoder == "linear":
            self.peak_projection = nn.Sequential( nn.Linear(peak_dim, peak_hidden_dim),
                                                  make_norm(norm_type, peak_hidden_dim),
                                                  make_activation(activation))

        elif peak_encoder == "fourier":
            self.mz_encoding_dim = peak_hidden_dim
            self.peak_extra_dim = max(peak_dim - 1, 0)
            if self.mz_encoding_dim % 2 != 0:
                raise ValueError("mz_encoding_dim must be even.")

            self.mz_mlp = nn.Sequential( nn.Linear(self.mz_encoding_dim, peak_hidden_dim),
                                         make_norm(norm_type, peak_hidden_dim),
                                         make_activation(activation),
                                         nn.Linear(peak_hidden_dim, peak_hidden_dim),
                                         make_norm(norm_type, peak_hidden_dim),
                                         make_activation(activation))

            peak_mlp_input_dim = peak_hidden_dim + self.peak_extra_dim

            self.peak_mlp = nn.Sequential(nn.Linear(peak_mlp_input_dim, peak_hidden_dim),
                                          make_norm(norm_type, peak_hidden_dim),
                                          make_activation(activation),
                                          nn.Dropout(dropout),
                                          nn.Linear(peak_hidden_dim, peak_hidden_dim),
                                          make_norm(norm_type, peak_hidden_dim),
                                          make_activation(activation))
        else:
            raise ValueError(f"Unknown peak_encoder={peak_encoder!r}. Use 'linear' or 'fourier'.")

        if self.use_resunits:
            self.peak_res_block = PeakResBlock( peak_hidden_dim=peak_hidden_dim,
                                                dropout=dropout,
                                                kernel_size=res_kernel_size,
                                                dilations=res_dilations,
                                                causal=res_causal)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, peak_hidden_dim))

        encoder_layer = CandyTransformerEncoderLayer( d_model=peak_hidden_dim,
                                                      nhead=heads,
                                                      ff_dim=ff_dim,
                                                      encoder_type=encoder_type,
                                                      norm_type=norm_type,
                                                      dropout=dropout,
                                                      activation=encoder_activation,
                                                      num_experts=num_experts,
                                                      moe_top_k=moe_top_k,
                                                      norm_first=norm_first)

        final_norm = make_norm(norm_type, peak_hidden_dim) if encoder_final_norm else None
        self.transformer = CandyTransformerEncoder(encoder_layer, num_layers=layers, norm=final_norm)

        self.type_emb = nn.Embedding(5, metadata_dim)
        self.mode_emb = nn.Embedding(3, metadata_dim)
        self.lc_emb = nn.Embedding(4, metadata_dim)
        self.modification_emb = nn.Embedding(4, metadata_dim)
        self.trap_emb = nn.Embedding(5, metadata_dim)

        self.prec_block = nn.Sequential(nn.Linear(input_precursor_dim, metadata_dim),
                                        make_norm(norm_type, metadata_dim),
                                        make_activation(activation))

        self.rt_block = nn.Sequential(nn.Linear(1, metadata_dim),
                                      make_norm(norm_type, metadata_dim),
                                      make_activation(activation))

        combined_dim = peak_hidden_dim + 7 * metadata_dim

        self.classifier_rep = nn.Sequential(nn.Linear(combined_dim, 1024),
                                            make_norm(norm_type, 1024),
                                            make_activation(activation),
                                            nn.Dropout(dropout),
                                            nn.Linear(1024, 512),
                                            make_norm(norm_type, 512),
                                            make_activation(activation),
                                            nn.Dropout(dropout))

        self.classifier_out = nn.Linear(512, num_classes)

    def encode_mz(self, mz):
        half_dim = self.mz_encoding_dim // 2
        wavelengths = (self.lambda_min * (self.lambda_max / self.lambda_min) **
                       (torch.arange(half_dim, dtype=mz.dtype, device=mz.device) / max(half_dim - 1, 1)))
        angles = 2.0 * torch.pi * mz / wavelengths.view(1, 1, half_dim)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def make_peak_embedding(self, peak_list):
        if self.peak_encoder == "linear":
            return self.peak_projection(peak_list)
        if self.peak_encoder == "fourier":
            mz = peak_list[..., 0:1]
            mz_encoded = self.encode_mz(mz)
            mz_embedding = self.mz_mlp(mz_encoded)
            if self.peak_extra_dim > 0:
                peak_extra = peak_list[..., 1:1 + self.peak_extra_dim]
                return self.peak_mlp(torch.cat([mz_embedding, peak_extra], dim=-1))
            return self.peak_mlp(mz_embedding)
        raise ValueError(f"Unknown peak_encoder={self.peak_encoder!r}.")

    def get_aux_loss(self):
        aux_loss = self.transformer.get_aux_loss()
        if aux_loss is None:
            return torch.tensor(0.0, device=self.cls_token.device, dtype=self.cls_token.dtype)
        return aux_loss
    def forward(self, peak_list, peak_padding_mask, precursor, glycan_type, rt, mode, lc, modification, trap, rep=False):
        batch_size = peak_list.size(0)
        peak_features = self.make_peak_embedding(peak_list)
        if self.use_resunits:
            peak_features = self.peak_res_block(peak_features,peak_padding_mask=peak_padding_mask)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        peak_features = torch.cat([cls_token, peak_features], dim=1)
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=peak_padding_mask.device)
        transformer_padding_mask = torch.cat([cls_padding, peak_padding_mask], dim=1)
        peak_features = self.transformer(peak_features,src_key_padding_mask=transformer_padding_mask)
        spectrum_rep = peak_features[:, 0, :]
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_block(precursor)
        rt = self.rt_block(rt)
        comb = torch.cat([spectrum_rep, precursor, glycan_type, rt, mode, lc, modification, trap], dim=1)
        comb_rep = self.classifier_rep(comb)
        comb = self.classifier_out(comb_rep)
        if rep:
            return comb, comb_rep
        return comb
