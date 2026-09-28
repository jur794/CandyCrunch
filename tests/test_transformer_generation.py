import numpy as np
import torch

from candycrunch.model import CandyCrunch_Transformer, SeqSimpleDataset


class _TinyTokenizer:
    special_tokens = {"pad_token": "<pad>"}
    vocab = {"<pad>": 0}

    def encode(self, strings):
        assert strings == ["example"]
        return {"token_ids": torch.tensor([[1, 3, 4, 2, 0]])}


def test_transformer_uses_shared_sequence_dataset_and_causal_decoder():
    features = [(
        np.zeros(4, dtype=np.float32),
        np.array([[100.0, 0.5], [200.0, 0.3], [0.0, 0.0]], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        1, 0.5, 1, 0, 0, 0,
    )]
    dataset = SeqSimpleDataset(features, [0], ["example"], _TinyTokenizer())
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))

    torch.manual_seed(0)
    model = CandyCrunch_Transformer(
        vocab_size=8, input_precursor_dim=2, heads=2, layers=1, ff_dim=32,
        peak_hidden_dim=16, metadata_dim=4, d_model=16, dec_heads=2,
        dec_layers=1, dec_ff_dim=32, dropout=0.0, dec_dropout=0.0,
        max_target_len=dataset.target_len,
    ).eval()

    spectrum, precursor = batch[1], batch[3]
    metadata = (batch[4], batch[5], batch[6], batch[7], batch[8], batch[9])
    input_ids, padding_mask, labels = batch[11], batch[12], batch[13]
    memory, memory_mask = model.encode(spectrum, precursor, *metadata)
    assert memory.shape == (1, 5, 16)  # CLS, three peaks, metadata
    assert memory_mask.tolist() == [[False, False, False, True, False]]
    assert labels.tolist() == [[3, 4, 2]]

    logits = model(spectrum, precursor, *metadata, input_ids, padding_mask)
    changed_ids = input_ids.clone()
    changed_ids[0, -1] = 5
    changed_logits = model(spectrum, precursor, *metadata, changed_ids, padding_mask)
    assert logits.shape == (1, 3, 8)
    torch.testing.assert_close(logits[:, :-1], changed_logits[:, :-1])

    logits.sum().backward()
    assert model.fc_out.weight.grad is not None
    assert model.memory_proj.weight.grad is not None
