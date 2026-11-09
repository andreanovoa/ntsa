"""Tests for `ntsa.fields.FieldSeries` on synthetic fields with known statistics."""

import json

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ntsa.fields import FieldSeries  # noqa: E402


def travelling_wave(nt=600, nx=64, k=3, c=0.5, dt=0.05, noise=0.0, seed=0):
    """u(t, x) = sin(k x - c k t) + noise on [0, 2 pi)."""
    x = np.arange(nx) * 2 * np.pi / nx
    t = np.arange(nt)[:, None] * dt
    u = np.sin(k * x[None] - c * k * t)
    return u + noise * np.random.default_rng(seed).standard_normal(u.shape), x[1] - x[0], dt


def test_wave_1d_statistics():
    u, dx, dt = travelling_wave()
    r = FieldSeries(u, dt=dt, dx=dx).analyze()
    assert r['rms'].shape == (64, 1)
    assert np.allclose(r['rms'], 1 / np.sqrt(2), atol=0.02)
    assert np.allclose(r['mean'], 0, atol=0.05)
    assert np.argmax(r['E']) == 2                              # energy at k = 3 (index 2 of k = 1, 2, ...)
    assert r['pod_n99'] == 2                                   # one travelling wave = one pair of POD modes
    assert np.isclose(r['f_peak'], 1.5 / (2 * np.pi), rtol=0.1)   # c k / 2 pi
    assert r['acf_lag1'] > 0.9


def test_energy_spectrum_sums_to_energy():
    u, dx, dt = travelling_wave(noise=0.3)
    fs = FieldSeries(u, dt=dt, dx=dx)
    r = fs.analyze()
    dk = r['k'][0]
    assert np.isclose((r['E'] * dk).sum(), 0.5 * np.mean(r['rms'] ** 2), rtol=1e-6)


def test_2d_components_and_shuffle_check(tmp_path):
    rng = np.random.default_rng(1)
    nt, ny, nx = 300, 24, 32
    y, x = np.meshgrid(np.arange(ny) * 2 * np.pi / ny, np.arange(nx) * 2 * np.pi / nx, indexing='ij')
    t = np.arange(nt)[:, None, None] * 0.05
    uu = np.cos(2 * x - t) * np.cos(2 * y)
    vv = np.sin(2 * y + 2 * t) * np.cos(3 * x)
    u = np.stack([uu, vv], axis=-1)
    fs = FieldSeries(u, dt=0.05, dx=(2 * np.pi / ny, 2 * np.pi / nx), components=('u', 'v'), label='synthetic')
    r = fs.analyze()
    assert r['pod_modes'].shape == (4, ny, nx, 2)
    assert r['pod_n99'] == 4
    assert not any('time-resolved' in c or 'box' in c for c in r['checks'])
    page = fs.report(tmp_path / 'report')
    assert (tmp_path / 'report' / 'animation.gif').exists()
    stats = json.loads((tmp_path / 'report' / 'stats.json').read_text())
    assert stats['shape (Nt, space, components)'] == [nt, [ny, nx], ['u', 'v']]
    assert 'data:image/png;base64' in open(page).read()

    shuffled = FieldSeries(u[rng.permutation(nt)], dt=0.05, components=('u', 'v')).analyze()
    assert any('not time-resolved' in c for c in shuffled['checks'])


def test_nonperiodic_and_transient_checks():
    u, dx, dt = travelling_wave(nt=2000, noise=0.05)
    u = u * np.where(np.arange(2000) < 400, np.linspace(0.05, 1, 2000)[:, None] ** 0, 1)[:, :1]
    growth = np.minimum(1.0, 0.05 + np.arange(2000) / 400)[:, None]
    r = FieldSeries(u * growth, dt=dt, dx=dx, periodic=False).analyze()
    assert r['i_stationary'] > 0
    assert any('transient' in c for c in r['checks'])


def test_shape_validation():
    with pytest.raises(ValueError):
        FieldSeries(np.zeros((10, 4, 4, 4, 4)), dt=1.0)
    with pytest.raises(ValueError):
        FieldSeries(np.zeros((10, 8, 3)), dt=1.0, components=('u', 'v'))


def test_reference_overlay(tmp_path):
    u, dx, dt = travelling_wave(noise=0.1)
    fs = FieldSeries(u, dt=dt, dx=dx)
    fs.add_reference('E', [1, 2, 3, 4], [1e-3, 1e-2, 1.0, 1e-2], 'Author et al. (2020)')
    with pytest.raises(ValueError):
        fs.add_reference('nonsense', [0], [0], 'x')
    fs.analyze()
    fig = fs.plot_statistics()
    labels = [ln.get_label() for ax in fig.axes for ln in ax.get_lines()]
    assert 'Author et al. (2020)' in labels
    page = open(fs.report(tmp_path / 'r', gif=False)).read()
    assert 'Author et al. (2020)' in page
