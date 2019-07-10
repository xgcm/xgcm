import numpy as np
from xarray import Dataset, DataArray

# from xray.core.ops import allclose_or_equiv
import pytest

# skip everythong
pytestmark = pytest.mark.xfail(True, reason="deprecated")

try:
    from xgcm import GCMDataset
except ImportError:
    # this import syntax is old
    pass


@pytest.fixture
def test_dataset():
    # need to create all the dimensions that GCMDataset likes
    # oceanic parameters, cartesian coordinates, doubly periodic
    H = 5000.0
    Lx = 4e6
    Ly = 3e6
    Nz = 10
    Nx = 25
    Ny = 20
    dz = H / Nz
    dx = Lx / Nx
    dy = Ly / Ny

    ds = Dataset()
    ds.attrs["H"] = H
    ds.attrs["Lx"] = Lx
    ds.attrs["Ly"] = Ly
    ds.attrs["Nz"] = Nz
    ds.attrs["Nx"] = Nx
    ds.attrs["Ny"] = Ny
    ds.attrs["dz"] = dz
    ds.attrs["dx"] = dx
    ds.attrs["dy"] = dy
    # vertical grid
    ds["Z"] = ("Z", dz / 2 + dz * np.arange(Nz))
    ds["Zp1"] = ("Zp1", dz * np.arange(Nz + 1))
    ds["Zl"] = ("Zl", dz * np.arange(Nz))
    ds["Zu"] = ("Zu", dz + dz * np.arange(Nz))
    # vertical spacing
    ds["drF"] = ("Z", np.full(Nz, dz))
    ds["drC"] = ("Zp1", np.hstack([dz / 2, np.full(Nz - 1, dz), dz / 2]))
    # horizontal grid
    ds["X"] = ("X", dx / 2 + dx * np.arange(Nx))
    ds["Xp1"] = ("Xp1", dx * np.arange(Nx))
    ds["Y"] = ("Y", dy / 2 + dy * np.arange(Ny))
    ds["Yp1"] = ("Yp1", dy * np.arange(Ny))
    xc, yc = np.meshgrid(ds.X, ds.Y)
    xg, yg = np.meshgrid(ds.Xp1, ds.Yp1)
    ds["XC"] = (("Y", "X"), xc)
    ds["YC"] = (("Y", "X"), yc)
    ds["XG"] = (("Yp1", "Xp1"), xg)
    ds["YG"] = (("Yp1", "Xp1"), yg)
    # horizontal spacing
    ds["dxC"] = (("Y", "Xp1"), np.full((Ny, Nx), dx))
    ds["dyC"] = (("Yp1", "X"), np.full((Ny, Nx), dy))
    ds["dxG"] = (("Yp1", "X"), np.full((Ny, Nx), dx))
    ds["dyG"] = (("Y", "Xp1"), np.full((Ny, Nx), dx))

    return ds


# class TestGCMDataset(unittest.TestCase):


def test_create_gcm_dataset(test_dataset):
    ds = test_dataset
    gcm = GCMDataset(ds)
    # should fail if any of the variables is missing
    for v in ds:
        with pytest.raises(KeyError):
            gcm = GCMDataset(ds.drop(v))


def test_vertical_derivatives(test_dataset):
    ds = test_dataset
    H = ds.attrs["H"]
    dz = ds.attrs["dz"]

    # vertical function of z at cell interface
    f = np.sin(np.pi * ds.Zp1.values / H)
    ds["f"] = (("Zp1"), f)
    ds["fl"] = ("Zl", f[:-1])
    # TODO: build in negative sign logic more carefully
    df = -np.diff(f)
    ds["df"] = ("Z", df)
    fill_value = 0.0
    ds["dfl"] = ("Z", np.hstack([df[:-1], f[-2] - fill_value]))
    ds["dfdz"] = ds["df"] / dz
    ds["dfldz"] = ds["dfl"] / dz

    # vertical function at cell center
    g = np.sin(np.pi * ds.Z.values / H)
    ds["g"] = ("Z", g)
    dg = -np.diff(g)
    dsdg = DataArray(dg, {"Zp1": ds.Zp1[1:-1]}, "Zp1")
    dsdgdf = dsdg / dz

    gcm = GCMDataset(ds)
    gcm_df = gcm.diff_zp1_to_z(ds.f)
    assert gcm_df.equals(ds.df), (gcm_df, ds.df)
    gcm_dfdz = gcm.derivative_zp1_to_z(ds.f)
    assert gcm_dfdz.equals(ds.dfdz), (gcm_dfdz, ds.dfdz)
    gcm_dfl = gcm.diff_zl_to_z(ds.fl, fill_value)
    assert gcm_dfl.equals(ds.dfl), (gcm_dfl, ds.dfl)
    gcm_dfldz = gcm.derivative_zl_to_z(ds.fl, fill_value)
    assert gcm_dfldz.equals(ds.dfldz), (gcm_dfldz, ds.dfldz)
    gcm_dg = gcm.diff_z_to_zp1(ds.g)
    assert gcm_dg.equals(dsdg), (gcm_dg, dsdg)
    gcm_dgdf = gcm.derivative_z_to_zp1(ds.g)
    assert gcm_dgdf.equals(dsdgdf), (gcm_dgdf, dsdgdf)


def test_vertical_integral(test_dataset):
    ds = test_dataset
    H = ds.attrs["H"]
    dz = ds.attrs["dz"]

    f = np.sin(np.pi * ds.Z.values / H)
    ds["f"] = (("Z"), f)
    ds["fint"] = (f * dz).sum()
    ds["favg"] = ds["fint"] / H

    gcm = GCMDataset(ds)
    gcm_fint = gcm.integrate_z(ds.f)
    assert gcm_fint.equals(ds.fint), (gcm_fint, ds.fint)
    gcm_favg = gcm.integrate_z(ds.f, average=True)
    assert gcm_favg.equals(ds.favg), (gcm_favg, ds.favg)


def test_horizontal_derivatives(test_dataset):
    ds = test_dataset
    dx = ds.attrs["dx"]
    dy = ds.attrs["dy"]
    Lx = ds.attrs["Lx"]
    Ly = ds.attrs["Ly"]

    # perdiodic function of Xp1
    f = np.sin(np.pi * ds.Xp1.values / Lx)
    ds["f"] = ("Xp1", f)
    ds["df"] = ("X", np.roll(f, -1) - f)
    ds["dfdx"] = ds.df / dx
    # periodic function of Yp1
    g = np.cos(np.pi * ds.Yp1.values / Ly)
    ds["g"] = ("Yp1", g)
    ds["dg"] = ("Y", np.roll(g, -1) - g)
    ds["dgdy"] = ds.dg / dy

    gcm = GCMDataset(ds)
    gcm_df = gcm.diff_xp1_to_x(ds.f)
    assert gcm_df.equals(ds.df), (gcm_df, ds.df)
    gcm_dg = gcm.diff_yp1_to_y(ds.g)
    assert gcm_dg.equals(ds.dg), (gcm_dg, ds.dg)
