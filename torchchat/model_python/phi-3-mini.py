
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM # , AutoTokenizer, pipeline


def model_builder(**config) -> nn.Module:
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    return model


