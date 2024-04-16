# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gguf
import torch
from quantize import group_dequantize_tensor_from_qparams


def to_float(t: gguf.gguf_reader.ReaderTensor):
    """
    Unpack and dequantize GGUF tensor to torch tensor of type torch.float32.
    """

    # All other weights are dequantized to float
    if t.tensor_type == gguf.GGMLQuantizationType.Q4_0:
        return group_dequantize_tensor_from_qparams(
            *Q4_0.unpack(t), Q4_0.n_bit, Q4_0.groupsize
        ).to(torch.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.Q6_K:
        return group_dequantize_tensor_from_qparams(
            *Q6_K.unpack(t), Q6_K.n_bit, Q6_K.groupsize
        ).to(torch.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.F16:
        return F16.unpack(t).to(torch.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.F32:
        return F32.unpack(t).to(torch.float32)
    else:
        raise ValueError(f"Unsupported tensor type {t.tensor_type}")


def test_by_to_float(source_file: str, target_file: str) -> None:
    """
    Tests methods in this file by using the to_float method, and comparing with a correct
    reference.  Raises error if there is a mismatch.

    In more detail, a GGUF source_file with various GGUF tensor types is parsed, and these
    tensors are converted with to_float.  These are then compared against a GGUF target_file.
    The target GGUF file must only contain F32 tensors, and should be generated by a method
    that is known to be correct.
    """

    gguf_sources = {t.name: t for t in gguf.GGUFReader(source_file, "r").tensors}
    gguf_targets = {t.name: t for t in gguf.GGUFReader(target_file, "r").tensors}

    for t in gguf_targets.values():
        assert (
            t.tensor_type == gguf.GGMLQuantizationType.F32
        ), f"target_file must only contain F32 tensors, but found tensor {t.name} with type {repr(t.tensor_type)}."
    assert (
        gguf_sources.keys() == gguf_targets.keys()
    ), "source_file and target_file should have the same tensors (by name)"

    for k in gguf_sources:
        source = to_float(gguf_sources[k])
        target = to_float(gguf_targets[k])

        if not torch.allclose(source, target):
            print(
                f"After calling to_float on source tensor {k} of type {repr(gguf_sources[k].tensor_type)} it does not match its target."
            )
            print("First 5 elements of converted source: ", source.reshape(-1)[0:5])
            print("First 5 elements of target: ", target.reshape(-1)[0:5])
            raise AssertionError("found mismatch")

    print("All tensors match.")


class F16:
    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF F16 tensor.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.F16
        reversed_shape = gguf_tensor.shape[::-1]
        new_tensor = gguf_tensor.data.reshape(reversed_shape)
        return torch.from_numpy(new_tensor).to(torch.float16)


class F32:
    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF F32 tensor.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.F32
        reversed_shape = gguf_tensor.shape[::-1]
        new_tensor = gguf_tensor.data.reshape(reversed_shape)
        return torch.from_numpy(new_tensor).to(torch.float32)


class Q4_0:
    groupsize = 32
    n_bit = 4

    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF Q4_0 matrix of size (nr, nc) to q, s, and z that can be dequantized by:

        x = s(q - 8) + z (roughly, reshape is needed),

        where
        * q is an int4-valued tensor of shape (nr, nc) and type torch.int32
        * s is a torch.float32 tensor of shape (nr, -1) with one scale per group
        * z is a torch.float32 tensor of shape (nr, -1) with one zero per group

        Note that z is always zero because Q4_0 is a scale-only scheme.

        See https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h for definition of block_q4_0:

        #define QK4_0 32
        typedef struct {
            ggml_half d;           // delta
            uint8_t qs[QK4_0 / 2]; // nibbles / quants
        } block_q4_0;

        Also see dequantize_row_q4_0 in https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
        for how the block should be interpreted.
        """

        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0
        assert len(gguf_tensor.shape) == 2
        nc, nr = gguf_tensor.shape  # GGUF tensor has reversed shape

        QK4_0 = 32  # groupsize

        # Parse block_q4_0
        block_q4_0_size = int(2 + QK4_0 / 2)
        packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q4_0_size))
        assert packed.dtype == torch.uint8
        ng = packed.shape[0]  # number of groups/blocks

        curr = 0
        size = 2  # half size
        d = packed[:, curr : (curr + size)].contiguous()
        d = torch.tensor(d.untyped_storage(), dtype=torch.float16).reshape(ng, 1)
        curr += size

        size = int(QK4_0 / 2)
        qs = packed[:, curr : (curr + size)].contiguous()
        curr += size

        # Check we finished parsing
        assert curr == block_q4_0_size

        # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract 8
        x0 = qs & 0x0F
        x1 = qs >> 4

        int32_data = torch.cat([x0, x1], dim=1).to(torch.int32).reshape(ng, QK4_0)
        assert int32_data.dtype == torch.int32
        assert int32_data.min().item() >= 0
        assert int32_data.max().item() <= 2**4 - 1
        assert int32_data.shape == (ng, QK4_0)

        # Prepare for return
        q = int32_data.to(torch.int32).reshape(nr, nc)
        s = d.to(torch.float32).reshape(nr, -1)
        z = torch.zeros(s.shape).to(torch.float32)
        return q, s, z


class Q6_K:
    groupsize = 16
    n_bit = 6

    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF Q6_k matrix of size (nr, nc) to q, s, and z that can be dequantized by:

        x = s(q - 32) + z (roughly, reshape is needed),

        where
        * q is an int6-valued tensor of shape (nr, nc) and type torch.int32
        * s is a torch.float32 tensor of shape (nr, -1) with one scale per group
        * z is a torch.float32 tensor of shape (nr, -1) with one zero per group

        Note that z is always zero because Q6_k is a scale-only scheme.

        See https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h for definition of block_q6_K:

        // 6-bit quantization
        // weight is represented as x = a * q
        // 16 blocks of 16 elements each
        // Effectively 6.5625 bits per weight
        typedef struct {
            uint8_t ql[QK_K/2];      // quants, lower 4 bits
            uint8_t qh[QK_K/4];      // quants, upper 2 bits
            int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
            ggml_half d;             // super-block scale
        } block_q6_K;

        QK_K is 64 or 256 by compile flag in the code, but in GGUF it looks like only the 256-variant
        is used, see "import gguf; gguf.GGML_QUANT_SIZES".

        Also see dequantize_row_q6_K in https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
        for how this block should be interpreted.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
        assert len(gguf_tensor.shape) == 2
        nc, nr = gguf_tensor.shape  # GGUF tensor has reversed shape
        QK_K = 256

        # Parse block_q6_K
        block_q6_K_size = int(QK_K / 2 + QK_K / 4 + QK_K / 16 + 2)
        packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q6_K_size))
        assert packed.dtype == torch.uint8
        ng = packed.shape[0]  # number of groups/blocks

        curr = 0

        size = int(QK_K / 2)
        ql = packed[:, curr : (curr + size)].contiguous()
        assert ql.shape == (ng, 128)
        curr += size

        size = int(QK_K / 4)
        qh = packed[:, curr : (curr + size)].contiguous()
        assert qh.shape == (ng, 64)
        curr += size

        size = int(QK_K / 16)
        scales = packed[:, curr : (curr + size)].contiguous()
        scales = (
            torch.tensor(scales.untyped_storage(), dtype=torch.int8)
            .reshape(ng, int(QK_K / 16))
            .to(torch.float32)
        )
        curr += size

        size = 2  # half size
        d = packed[:, curr : (curr + size)].contiguous()
        d = (
            torch.tensor(d.untyped_storage(), dtype=torch.float16)
            .reshape(ng, 1)
            .to(torch.float32)
        )
        curr += size

        # Check we finished parsing
        assert curr == block_q6_K_size

        # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract 32
        q1 = (ql[:, 0:32] & 0xF) | (((qh[:, 0:32] >> 0) & 3) << 4)
        q2 = (ql[:, 32:64] & 0xF) | (((qh[:, 0:32] >> 2) & 3) << 4)
        q3 = (ql[:, 0:32] >> 4) | (((qh[:, 0:32] >> 4) & 3) << 4)
        q4 = (ql[:, 32:64] >> 4) | (((qh[:, 0:32] >> 6) & 3) << 4)

        q5 = (ql[:, 64:96] & 0xF) | (((qh[:, 32:64] >> 0) & 3) << 4)
        q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:64] >> 2) & 3) << 4)
        q7 = (ql[:, 64:96] >> 4) | (((qh[:, 32:64] >> 4) & 3) << 4)
        q8 = (ql[:, 96:128] >> 4) | (((qh[:, 32:64] >> 6) & 3) << 4)

        q = torch.cat([q1, q2, q3, q4, q5, q6, q7, q8], dim=1).to(torch.int32)
        assert q.shape == (ng, QK_K)
        assert q.min().item() >= 0
        assert q.max().item() <= 2**6 - 1

        # Unpack scales
        s1 = d * torch.cat(
            [scales[:, 0].reshape(-1, 1), scales[:, 1].reshape(-1, 1)], dim=1
        )
        s2 = d * torch.cat(
            [scales[:, 2].reshape(-1, 1), scales[:, 3].reshape(-1, 1)], dim=1
        )
        s3 = d * torch.cat(
            [scales[:, 4].reshape(-1, 1), scales[:, 5].reshape(-1, 1)], dim=1
        )
        s4 = d * torch.cat(
            [scales[:, 6].reshape(-1, 1), scales[:, 7].reshape(-1, 1)], dim=1
        )

        s5 = d * torch.cat(
            [scales[:, 8].reshape(-1, 1), scales[:, 9].reshape(-1, 1)], dim=1
        )
        s6 = d * torch.cat(
            [scales[:, 10].reshape(-1, 1), scales[:, 11].reshape(-1, 1)], dim=1
        )
        s7 = d * torch.cat(
            [scales[:, 12].reshape(-1, 1), scales[:, 13].reshape(-1, 1)], dim=1
        )
        s8 = d * torch.cat(
            [scales[:, 14].reshape(-1, 1), scales[:, 15].reshape(-1, 1)], dim=1
        )
        s = torch.cat([s1, s2, s3, s4, s5, s6, s7, s8], dim=1)
        assert s.shape == (ng, 16)

        # Prepare for return
        q = q.to(torch.int32).reshape(nr, nc)
        s = s.reshape(nr, -1)
        z = torch.zeros(s.shape).to(torch.float32)

        return q, s, z
