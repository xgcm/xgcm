# python 3 compatiblity
from __future__ import print_function, division

# make some functions for taking divergence
from dask import array as da
import xray
import numpy as np


def _append_to_name(array, append):
    try:
        return array.name + "_" + append
    except TypeError:
        return append


class GCMDataset(object):
    """Representation of GCM (General Circulation Model) output data, numerical
    grid information, and operations related to finite-volume analysis.
    """

    # without these variables in the dataset, can't function
    needed_vars = [
        "Z",
        "Zp1",
        "Zl",
        "Zu",
        "X",
        "Xp1",
        "Y",
        "Yp1",
        "XC",
        "YC",
        "XG",
        "YG",
        "drF",
        "drC",
        "dxC",
        "dxG",
        "dyC",
        "dyG",
    ]

    def __init__(self, ds):
        """Initialize GCM object.

        Parameters
        ----------
        ds : xray Dataset
        """

        # check that needed variables are present
        for v in self.needed_vars:
            if v not in ds:
                raise KeyError("Needed variable %s not found in dataset" % v)

        self.ds = ds

    # silly functions

    def _get_coords_from_dims(self, dims, replace=None):
        """Utility function for quickly fetching coordinates from parent
        dataset.
        """
        dims = list(dims)
        if replace:
            for k in replace:
                dims[dims.index(k)] = replace[k]
        return {dim: self.ds[dim] for dim in dims}, dims

    def _get_hfac_for_array(self, array):
        """Figure out the correct hfac given array dimensions."""
        hfac = None
        if "X" in array.dims and "Y" in array.dims and "HFacC" in self.ds:
            hfac = self.ds.HFacC
        if "Xp1" in array.dims and "Y" in array.dims and "HFacW" in self.ds:
            hfac = self.ds.HFacW
        if "X" in array.dims and "Yp1" in array.dims and "HFacW" in self.ds:
            hfac = self.ds.HFacS
        return hfac

    ### Vertical Differences, Derivatives, and Interpolation ###

    def pad_zl_to_zp1(self, array, fill_value=0.0, zlname="Zl", zp1name="Zp1"):
        """Pad an array located at zl points such that it is located at
        zp1 points. An additional fill value is required for the bottom point.

        Parameters
        ----------
        array : xray DataArray
            The array to difference. Must have the coordinate zp1.
        fill_value : number, optional
            The value to be used at the bottom point.
        zlname : str, optional
            The variable name for the zl point
        zp1name : str, optional
            The variable name for the zp1 point

        Returns
        -------
        padded : xray DataArray
            Padded array with vertical coordinate zp1.
        """
        coords, dims = self._get_coords_from_dims(array.dims)
        zdim = dims.index(zlname)
        # shape of the new array to concat at the bottom
        shape = list(array.shape)
        shape[zdim] = 1
        # replace Zl with the bottom level
        coords[zlname] = np.atleast_1d(self.ds[zp1name][-1].data)
        # an array of zeros at the bottom
        # need different behavior for numpy vs dask
        if array.chunks:
            chunks = list(array.data.chunks)
            chunks[zdim] = (1,)
            zarr = fill_value * da.ones(shape, dtype=array.dtype, chunks=chunks)
            zeros = xray.DataArray(zarr, coords, dims).chunk()
        else:
            zarr = np.zeros(shape, array.dtype)
            zeros = xray.DataArray(zarr, coords, dims)
        newarray = xray.concat([array, zeros], dim=zlname).rename({zlname: zp1name})
        if newarray.chunks:
            # this assumes that there was only one chunk in the vertical to begin with
            # how can we do that better
            return newarray.chunk({zp1name: len(newarray[zp1name])})
        else:
            return newarray

    def diff_zp1_to_z(self, array, zname="Z", zp1name="Zp1"):
        """Take the vertical difference of an array located at zp1 points, resulting
        in a new array at z points.

        Parameters
        ----------
        array : xray DataArray
            The array to difference. Must have the coordinate zp1.
        zname : str, optional
            The variable name for the z point
        zp1name : str, optional
            The variable name for the zp1 point

        Returns
        -------
        diff : xray DataArray
            A new array with vertical coordinate z.
        """
        a_up = array.isel(**{zp1name: slice(None, -1)})
        a_dn = array.isel(**{zp1name: slice(1, None)})
        a_diff = a_up.data - a_dn.data
        # dimensions and coords of new array
        coords, dims = self._get_coords_from_dims(array.dims, replace={zp1name: zname})
        return xray.DataArray(
            a_diff, coords, dims, name=_append_to_name(array, "diff_zp1_to_z")
        )

    def diff_zl_to_z(self, array, fill_value=0.0):
        """Take the vertical difference of an array located at zl points, resulting
        in a new array at z points. A fill value is required to provide the bottom
        boundary condition for ``array``.

        Parameters
        ----------
        array : xray DataArray
            The array to difference. Must have the coordinate zl.
        fill_value : number, optional
            The value to be used at the bottom point. The default (0) is the
            appropriate choice for vertical fluxes.

        Returns
        -------
        diff : xray DataArray
            A new array with vertical coordinate z.
        """
        array_zp1 = self.pad_zl_to_zp1(array, fill_value)
        array_diff = self.diff_zp1_to_z(array_zp1)
        return array_diff.rename(_append_to_name(array, "_diff_zl_to_z"))

    def diff_z_to_zp1(self, array):
        """Take the vertical difference of an array located at z points, resulting
        in a new array at zp1 points, but missing the upper and lower point.

        Parameters
        ----------
        array : xray DataArray
            The array to difference. Must have the coordinate z.

        Returns
        -------
        diff : xray DataArray
            A new array with vertical coordinate zp1.
        """
        a_up = array.isel(Z=slice(None, -1))
        a_dn = array.isel(Z=slice(1, None))
        a_diff = a_up.data - a_dn.data
        # dimensions and coords of new array
        coords, dims = self._get_coords_from_dims(array.dims, replace={"Z": "Zp1"})
        # trim vertical
        coords["Zp1"] = coords["Zp1"][1:-1]
        return xray.DataArray(
            a_diff, coords, dims, name=_append_to_name(array, "diff_z_to_zp1")
        )

    def derivative_zp1_to_z(self, array):
        """Take the vertical derivative of an array located at zp1 points, resulting
        in a new array at z points.

        Parameters
        ----------
        array : xray DataArray
            The array to differentiate. Must have the coordinate zp1.

        Returns
        -------
        deriv : xray DataArray
            A new array with vertical coordinate z.
        """
        a_diff = self.diff_zp1_to_z(array)
        dz = self.ds.drF
        return a_diff / dz

    def derivative_zl_to_z(self, array, fill_value=0.0):
        """Take the vertical derivative of an array located at zl points, resulting
        in a new array at z points. A fill value is required to provide the bottom
        boundary condition for ``array``.

        Parameters
        ----------
        array : xray DataArray
            The array to differentiate. Must have the coordinate zl.
        fill_value : number, optional
            The assumed value at the bottom point. The default (0) is the
            appropriate choice for vertical fluxes.

        Returns
        -------
        deriv : xray DataArray
            A new array with vertical coordinate z.
        """

        a_diff = self.diff_zl_to_z(array, fill_value)
        dz = self.ds.drF
        return a_diff / dz

    def derivative_z_to_zp1(self, array):
        """Take the vertical derivative of an array located at z points, resulting
        in a new array at zp1 points, but missing the upper and lower point.

        Parameters
        ----------
        array : xray DataArray
            The array to differentiate. Must have the coordinate z.

        Returns
        -------
        diff : xray DataArray
            A new array with vertical coordinate zp1.
        """
        a_diff = self.diff_z_to_zp1(array)
        dz = self.ds.drC[1:-1]
        return a_diff / dz

    ### Vertical Integrals ###
    # if the array to integrate is 1D or 2D, don't multiply by hFac
    # but what if it is 3D, how do we decide what to do?
    # what if points are missing? xray should take care of that
    # how do we pick which hFac to use? look at dims
    def integrate_z(self, array, average=False):
        """Integrate ``array`` in vertical dimension, accounting for vertical
        grid geometry.

        Parameters
        ----------
        array : xray DataArray
            The array to integrate. Must have the dimension Z.
        average : bool, optional
            If ``True``, return an average instead of an integral.

        Returns
        -------
        integral : xray DataArray
            The vertical integral of ``array``.
        """
        if not "Z" in array.dims:
            raise ValueError("Can only integrate arrays on Z grid")
        dz = self.ds.drF
        # look at horizontal dimensions and try to find an hfac
        hfac = self._get_hfac_for_array(array)
        if hfac is not None:
            # brodcast hfac against dz
            dz *= hfac
        a_int = (array * dz).sum(dim="Z")
        if average:
            return a_int / dz.sum(dim="Z")
        else:
            return a_int

    ### Horizontal Differences, Derivatives, and Interpolation ###

    # doesn't actually need parent ds
    # this could go in xray
    def roll(self, array, n, dim):
        """Clone of numpy.roll for xray DataArrays."""
        left = array.isel(**{dim: slice(None, -n)})
        right = array.isel(**{dim: slice(-n, None)})
        return xray.concat([right, left], dim=dim)

    def diff_xp1_to_x(self, array):
        """Difference DataArray ``array`` in the x direction.
        Assumes that ``array`` is located at the xp1 point."""
        left = array
        right = self.roll(array, -1, "Xp1")
        if array.chunks:
            right = right.chunk(array.chunks)
        diff = right.data - left.data
        coords, dims = self._get_coords_from_dims(array.dims, replace={"Xp1": "X"})
        return xray.DataArray(diff, coords, dims).rename(
            _append_to_name(array, "diff_xp1_to_x")
        )

    def diff_yp1_to_y(self, array):
        """Difference DataArray ``array`` in the y direction.
        Assumes that ``array`` is located at the yp1 point."""
        left = array
        right = self.roll(array, -1, "Yp1")
        if array.chunks:
            right = right.chunk(array.chunks)
        diff = right.data - left.data
        coords, dims = self._get_coords_from_dims(array.dims, replace={"Yp1": "Y"})
        return xray.DataArray(diff, coords, dims).rename(
            _append_to_name(array, "_diff_yp1_to_y")
        )
