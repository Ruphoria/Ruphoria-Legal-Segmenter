"""Fetch Ulysses pretrained segmenter models."""
import os

import buscador


__all__ = [
    "download_model",
    "get_model_uri_if_local_file",
]


def download_model(model_name: str, output_dir: str, show_progress_bar: bool = True) -> bool:
    """Fetch a pretrained model or tokenizer.

    Parameters
    ----------
    model_name : str
        Model to fetch.

    output_dir : str
        Directory to save downloaded model.

    show_progress_bar : bool, default=True
        If True, display download progress bar.

    Returns
    -------
    has_succeed : bool
        True if download succeed, or a cached local file was found.
    """
    try:
        download_has_succeed = buscador.download_resource(
            task_name="legal_text_segmentation",
            resource_name=model_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=True,
            clean_compressed_files=True,
            check_resource_hash=True,
            timeout_limit_seconds=60,
        )
        return bool(download_has_succeed)

    except ValueError:
        return False


def get_model_uri_if_local_file(
    model_name: str, download_dir: str, file_extension: str = ""
) -> str:
    """Build a full URI for a downloaded model if found locally.

    Parameters
    ----------
    model_name : str
        Model name to search for.

    download_dir : str
        Where to find model.

    file_extension : str, default=''
        File extension to append to URI. Unnecessary if `model_name` has the file extension,
        or the model has no file extension at all.

    Returns
    -------
    path : str
        Model URI if found locally, `model_name` otherwise.
    """
    uri_model = str(model_name).strip()
    uri_model = os.path.join(download_dir, ur