# -*- mode: python ; coding: utf-8 -*-
"""BlackTune PyInstaller spec file."""

a = Analysis(
    ['blacktune/main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'pyqtgraph',
        'scipy.signal',
        'scipy.fft',
        'scipy.special',
        'pandas',
        'orangebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BlackTune',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed
    icon=None,
)
