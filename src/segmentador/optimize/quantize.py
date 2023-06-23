"""Apply quantization and hardware-specific optimizations in segmenter models."""
import typing as t
import pickle
import os
import pathlib
import collections
import platform
import warnings
import random
import datetime
import shutil

import transformers
import torch
import torch.nn
import torch.onnx

from .. import _base
from .. import segmenter
from . import _optional_import_utils
from . import models


colorama = _optional_import_utils.load_optional_module("colorama")


__all__ = [
    "quantize_bert_model_as_onnx",
    "quantize_bert_model_as_torch",
    "quantize_lstm_model_as_onnx",
    "quantize_lstm_model_as_torch",
    "quantize_model",
]


class QuantizationOutputONNX(t.NamedTuple):
    """Output paths for quantization as ONNX format."""

    onnx_base_uri: str
    onnx_quantized_uri: str
    output_uri: str
    onnx_optimized_uri: t.Optional[str] = None


class QuantizationOutputTorch(t.NamedTuple):
    """Quantization output paths as Torch format."""

    output_uri: str


QuantizationOutput = t.Union[QuantizationOutputONNX, QuantizationOutputTorch]


def _build_onnx_default_uris(
    model_name: str,
    model_attributes: t.Dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
) -> QuantizationOutputONNX:
    """Build default URIs for quantized output in ONNX format."""
    if not intermediary_onnx_model_name:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        intermediary_onnx_model_name = f"segmenter_{attrs_to_name}_{model_name}_model"

    if not quantized_model_filename:
        quantized_model_filename = f"q_{intermediary_onnx_model_name}"

    if not intermediary_onnx_model_name.endswith(".onnx"):
        intermediary_onnx_model_name += ".onnx"

    if not quantized_model_filename.endswith(".onnx"):
        quantized_model_filename += ".onnx"

    pathlib.Path(quantized_model_dirpath).mkdir(exist_ok=True, parents=True)

    onnx_base_uri = os.path.join(quantized_model_dirpath, intermediary_onnx_model_name)
    onnx_quantized_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths_dict: t.Dict[str, str] = {
        "onnx_base_uri": onnx_base_uri,
        "onnx_quantized_uri": onnx_quantized_uri,
        "output_uri": onnx_quantized_uri,
    }

    paths = QuantizationOutputONNX(**paths_dict)

    all_path_set = {paths.onnx_base_uri, paths.onnx_quantized_uri}
    num_distinct_paths = len(all_path_set)

    if num_distinct_paths < 2:
        raise ValueError(
            f"{2 - num_distinct_paths} URI for ONNX models (including intermediary models) "
            "are the same, which will cause undefined behaviour while quantizing the model. "
            "Please provide distinct filenames for ONNX files."
        )

    return paths


def _build_torch_default_uris(
    model_name: str,
    model_attributes: t.Dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
) -> QuantizationOutputTorch:
    """Build default URIs for quantized output in Torch format."""
    if not quantized_model_filename:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        quantized_model_filename = f"q_segmenter_{attrs_to_name}_{model_name}_model.pt"

    pathlib.Path(quantized_model_dirpath).mkdir(exist_ok=True, parents=True)
    output_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths = QuantizationOutputTorch(output_uri=output_uri)

    return paths


def _gen_dummy_inputs_for_tracing(
    batch_size: int, vocab_size: int, seq_length: int
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dummy inputs for Torch JIT tracing."""
    dummy_input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_length), dtype=torch.long
    )
    dummy_attention_mask = torch.randint(
        low=0, high=2, size=(batch_size, seq_length), dtype=torch.long
    )
    dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    return dummy_input_ids, dummy_attention_mask, dummy_token_type_ids


def quantize_bert_model_as_onnx(
    model: segmenter.BERTSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    """Create a quantized BERTSegmenter model as ONNX format.

    Models created from this format can be loaded for inference as:

    >>> optimize.ONNXBERTSegmenter(  # doctest: +SKIP
    ...     uri_model='<quantized_model_uri>',
    ...     uri_tokenizer=...,
    ...     ...,
    ... )

    Parameters
    ----------
    model : segmenter.BERTSegmenter
        BERTSegmenter model to be quantized.

    quantized_model_filename : str or None, default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    intermediary_onnx_model_name : str or None, default=None
        Name to save intermediary model in ONNX format in `quantized_model_dirpath`. This
        transformation is necessary to perform all necessary optimization and quantization.
        If None, a name will be derived from `quantized_model_filename`.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.

    References
    ----------
    .. [1] Graph Optimizations in ONNX Runtime. Available at:
       https://onnxruntime.ai/docs/performance/graph-optimizations.html

    .. [2] ONNX Operator Schemas. Available at:
       https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    optimum_onnxruntime = _optional_import_utils.load_required_module("optimum.onnxruntime")

    model_config: transformers.BertConfig = model.model.config  # type: ignore
    is_pruned = bool(model_config.pruned_heads)

    if is_pruned:
        raise RuntimeError(
            "BERT with pruned attention heads will not work in ONNX format. Please use Torch "
            "format instead (with quantize_bert_model_as_torch(...) function or by using "
            "quantize_model(..., model_output_format='torch_jit')."
        )

    model_attributes: t.Dict[str