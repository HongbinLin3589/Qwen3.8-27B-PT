"""Keep the shared token/mask fingerprint compatible with an independent byte oracle."""

import hashlib
import struct

import pytest
import torch

from qwenpt.train.verl.template import tokenized_fingerprint


@pytest.mark.parametrize("input_dtype,input_format,mask_dtype,mask_format", [
    (torch.int64, "q", torch.int64, "q"),
    (torch.int32, "i", torch.int32, "i"),
    (torch.int64, "q", torch.bool, "?"),
    (torch.int64, "q", torch.float32, "f"),
    (torch.int64, "q", torch.float64, "d"),
])
def test_fingerprint_matches_original_dtype_bytes_for_strided_inputs(
    input_dtype, input_format, mask_dtype, mask_format,
):
    """Native struct bytes independently fix ordering, dtype and unshifted-mask semantics."""
    ids, mask = [1, 248319, -7, 0], [0, 1, 0, 1]
    input_ids = torch.tensor([value for item in ids for value in (item, 99)], dtype=input_dtype)[::2]
    loss_mask = torch.tensor([value for item in mask for value in (item, 0)], dtype=mask_dtype)[::2]
    assert not input_ids.is_contiguous() and not loss_mask.is_contiguous()
    if loss_mask.is_floating_point():
        loss_mask.requires_grad_()
    expected_bytes = struct.pack("=" + input_format * len(ids), *ids) + struct.pack("=" + mask_format * len(mask), *mask)
    expected = hashlib.sha256(expected_bytes).hexdigest()
    assert tokenized_fingerprint(input_ids, loss_mask) == expected
    assert input_ids.tolist() == ids and loss_mask.tolist() == mask
    assert tokenized_fingerprint(input_ids, loss_mask.roll(1)) != expected


def test_fingerprint_preserves_empty_and_unsupported_numpy_dtype_behavior():
    """The shared protocol adds neither nonempty validation nor silent dtype conversion."""
    empty = torch.empty(0, dtype=torch.int64)
    assert tokenized_fingerprint(empty, empty) == hashlib.sha256(b"").hexdigest()
    with pytest.raises(TypeError, match="BFloat16"):
        tokenized_fingerprint(torch.tensor([1]), torch.tensor([0], dtype=torch.bfloat16))
