@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  VAJRA — Fix NumPy cp314/cp311 mismatch + install all L2 dependencies
REM
REM  Problem: venv runs Python 3.11 but NumPy was installed by a Python 3.14
REM  interpreter somewhere, building cp314-win_amd64 binaries that 3.11
REM  cannot load.
REM
REM  Run from the proxy root:
REM    cd D:\denova\sandbox\proxy
REM    fix_env.bat
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo [1/5] Confirming Python version in venv...
venv\Scripts\python.exe --version
echo.

echo [2/5] Uninstalling broken NumPy (cp314 build)...
venv\Scripts\pip.exe uninstall numpy -y

echo.
echo [3/5] Reinstalling NumPy for Python 3.11 (cp311)...
venv\Scripts\pip.exe install "numpy>=1.26,<3" --force-reinstall

echo.
echo [4/5] Installing Layer 2 dependencies...
venv\Scripts\pip.exe install fastembed faiss-cpu

echo.
echo [5/5] Verifying imports...
venv\Scripts\python.exe -c "import numpy; print('numpy', numpy.__version__)"
venv\Scripts\python.exe -c "import fastembed; print('fastembed OK')"
venv\Scripts\python.exe -c "import faiss; print('faiss OK')"

echo.
echo ═══════════════════════════════════════════════════════════════════════
echo  Done. Run tests with:
echo    cd test
echo    ..\venv\Scripts\python.exe test_layer2.py
echo ═══════════════════════════════════════════════════════════════════════