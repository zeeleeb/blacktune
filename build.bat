@echo off
echo Building BlackTune...
cd /d %~dp0
venv\Scripts\pyinstaller --onefile --windowed ^
    --name BlackTune ^
    --icon=NUL ^
    --hidden-import=pyqtgraph ^
    --hidden-import=scipy.signal ^
    --hidden-import=scipy.fft ^
    --hidden-import=pandas ^
    --hidden-import=orangebox ^
    blacktune\main.py
echo.
if exist dist\BlackTune.exe (
    echo Build successful: dist\BlackTune.exe
    echo File size:
    for %%A in (dist\BlackTune.exe) do echo   %%~zA bytes
) else (
    echo Build FAILED
)
