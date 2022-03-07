from typing import Callable, Tuple, Mapping

import numpy as np


class Padder:
    """
    Internal class which can pad numpy arrays, or compose a numpy ufunc with a particular padding operation.

    One instance of the Padder class should be used for one padding operation, as its internals store the options chosen
    for that particular use.
    """
    grid: "Grid"
    boundary_width: ...  # TODO work out what all these types need to be
    boundary: ...
    fill_value: ...

    def __init__(self, grid, boundary_width, boundary, fill_value):
        self.grid = grid
        self.boundary_width = boundary_width
        self.boundary = boundary
        self.fill_value = fill_value

    def pad(self, *arrays: np.ndarray) -> Tuple[np.ndarray]:

        # do the padding
        padded_arrs = ...

        # merge any lonely chunks on either end created by padding
        if any(
                _has_chunked_core_dims(padded_arg, core_dims)
                for padded_arr, core_dims in zip(padded_arrs, in_core_dims)
        ):
            return self.merge_boundary_chunks(arrays)
        else:
            return arrays

    def merge_boundary_chunks(
        self,
        *arrs: np.ndarray,
        original_chunks: Mapping[int, Tuple[int, int]],
    ) -> Tuple[np.ndarray]:
        """Merge in any small floating chunks at the edges that were created by the padding operation"""
        ...

    def compose_with(self, func: Callable) -> Callable:
        """Return a new function which first pads its arguments and then applies `func` to them."""

        def pad_then_func(*args, **kwargs):
            padded_args = self.pad(*args)
            return func(*padded_args, **kwargs)

        return pad_then_func


### ignore past here for now


def _is_dim_chunked(a, dim):
    # TODO this func can't handle Datasets - it will error if you check multiple variables with different chunking
    return len(a.variable.chunksizes[dim]) > 1


def _has_chunked_core_dims(obj: xr.DataArray, core_dims: Sequence[str]) -> bool:
    # TODO what if only some of the core dimensions are chunked?
    return obj.chunks is not None and any(
        _is_dim_chunked(obj, dim) for dim in core_dims
    )


def _rechunk_to_merge_in_boundary_chunks(
    padded_args: Sequence[xr.DataArray],
    original_args: Sequence[xr.DataArray],
    boundary_width_real_axes: Mapping[str, Tuple[int, int]],
    grid: "Grid",
) -> List[xr.DataArray]:
    """Merges in any small floating chunks at the edges that were created by the padding operation"""

    rechunked_padded_args = []
    for padded_arg, original_arg in zip(padded_args, original_args):

        original_arg_chunks = original_arg.variable.chunksizes
        merged_boundary_chunks = _get_chunk_pattern_for_merging_boundary(
            grid,
            padded_arg,
            original_arg_chunks,
            boundary_width_real_axes,
        )
        rechunked_arg = padded_arg.chunk(merged_boundary_chunks)
        rechunked_padded_args.append(rechunked_arg)

    return rechunked_padded_args


def _get_chunk_pattern_for_merging_boundary(
    grid: "Grid",
    da: xr.DataArray,
    original_chunks: Mapping[str, Tuple[int, ...]],
    boundary_width_real_axes: Mapping[str, Tuple[int, int]],
) -> Mapping[str, Tuple[int, ...]]:
    """Calculates the pattern of chunking needed to merge back in small chunks left on boundaries after padding"""

    # Easier to work with width of boundaries in terms of str dimension names rather than int axis numbers
    boundary_width_dims = {
        _get_dim(grid, da, ax): width for ax, width in boundary_width_real_axes.items()
    }

    new_chunks: Dict[str, Tuple[int, ...]] = {}
    for dim, width in boundary_width_dims.items():
        lower_boundary_width, upper_boundary_width = boundary_width_dims[dim]

        new_chunks_along_dim: Tuple[int, ...]
        if len(original_chunks[dim]) == 1:
            # unpadded array had only one chunk, but padding has meant new array is extended
            original_array_length = original_chunks[dim][0]
            new_chunks_along_dim = (
                lower_boundary_width + original_array_length + upper_boundary_width,
            )
        else:
            first_chunk_width, *other_chunks_widths, last_chunk_width = original_chunks[
                dim
            ]
            new_chunks_along_dim = tuple(
                [
                    first_chunk_width + lower_boundary_width,
                    *other_chunks_widths,
                    last_chunk_width + upper_boundary_width,
                ]
            )
        new_chunks[dim] = new_chunks_along_dim

    return new_chunks