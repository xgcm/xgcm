import numpy as np

from xray import Dataset, DataArray
from xray.core.ops import allclose_or_equiv
from xgcm import GCMDataset

# better unitttest2?
import unittest

def create_test_dataset():
    # need to create all the dimensions that GCMDataset likes
    # oceanic parameters, cartesian coordinates, doubly periodic
    H = 5000.
    Lx = 4e6
    Ly = 3e6
    Nz = 10
    Nx = 25
    Ny = 20
    dz = H / Nx
    dx = Lx / Nx
    dy = Ly / Ny

    ds = Dataset()
    ds.attrs['H'] = H
    ds.attrs['Lx'] = Lx
    ds.attrs['Ly'] = Ly
    ds.attrs['Nz'] = Nz
    ds.attrs['Nx'] = Nx
    ds.attrs['Ny'] = Ny
    ds.attrs['dz'] = dz
    ds.attrs['dx'] = dx
    ds.attrs['dy'] = dy
    # vertical grid
    ds['Z'] = ('Z', dz/2 + dz*np.arange(Nz))
    ds['Zp1'] = ('Zp1', dz*np.arange(Nz+1))
    ds['Zl'] = ('Zl', dz*np.arange(Nz))
    ds['Zu'] = ('Zu', dz + dz*np.arange(Nz))
    # vertical spacing
    ds['drF'] = ('Z', np.full(Nz, dz))
    ds['drC'] = ('Zp1', np.hstack([dz/2, np.full(Nz-1, dz), dz/2]))
    # horizontal grid
    ds['X'] = ('X', dx/2 + dx*np.arange(Nx))
    ds['Xp1'] = ('Xp1', dx*np.arange(Nx))
    ds['Y'] = ('Y', dy/2 + dy*np.arange(Ny))
    ds['Yp1'] = ('Yp1', dy*np.arange(Ny))
    xc, yc = np.meshgrid(ds.X, ds.Y)
    xg, yg = np.meshgrid(ds.Xp1, ds.Yp1)
    ds['XC'] = (('Y','X'), xc)
    ds['YC'] = (('Y','X'), yc)
    ds['XG'] = (('Yp1','Xp1'), xg)
    ds['YG'] = (('Yp1','Xp1'), yg)
    # horizontal spacing
    ds['dxC'] = (('Y','Xp1'), np.full((Ny,Nx), dx))
    ds['dyC'] = (('Yp1','X'), np.full((Ny,Nx), dy))
    ds['dxG'] = (('Yp1','X'), np.full((Ny,Nx), dx))
    ds['dyG'] = (('Y','Xp1'), np.full((Ny,Nx), dx))

    return ds


class TestGCMDataset(unittest.TestCase):

    def test_create_gcm_dataset(self):
        ds = create_test_dataset()
        gcm = GCMDataset(ds)
        # should fail if any of the variables is missing
        for v in ds:
            with self.assertRaises(KeyError):
                gcm = GCMDataset(ds.drop(v))

    def test_vertical_derivatives(self):
        ds = create_test_dataset()
        H = ds.attrs['H']
        # vertical function of z at cell interface
        f = np.sin(np.pi * ds.Zp1.values / H)
        ds['f'] = (('Zp1'), f)
        # TODO: build in negative sign logic more carefully
        df = -np.diff(f)
        ds['df'] = ('Z', df)
        ds['dfdz'] = ds['df'] / ds.attrs['dz']
        # analytical first derivative at c points
        # too much precision to expect for finite difference
        #g = -np.pi/H * np.cos(np.pi * ds.Z / H)
        gcm = GCMDataset(ds)
        assert gcm.diff_zp1_to_z(ds.f).equals(ds.df)
        assert gcm.derivative_zp1_to_z(ds.f).equals(ds.dfdz)
