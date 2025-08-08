@echo off
REM Brain-Computer Compression Toolkit - Quick Setup and Validation Script (Windows)

echo ==========================================
echo Brain-Computer Compression Toolkit Setup
echo ==========================================

REM Check Python version
echo 🐍 Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo ⚠️  requirements.txt not found, installing basic dependencies...
    pip install numpy scipy matplotlib
)

REM Install EMG requirements if available
if exist "requirements-emg.txt" (
    echo 📦 Installing EMG requirements...
    pip install -r requirements-emg.txt
)

REM Install toolkit in development mode
echo 📦 Installing toolkit...
pip install -e .

echo.
echo ✅ Installation completed!
echo.

REM Run dependency check
echo 🔍 Checking dependencies...
cd tests
python run_tests.py --dependencies-only
if %errorlevel% neq 0 (
    echo ❌ Some dependencies missing
    pause
    exit /b 1
)
echo ✅ All dependencies available

echo.
echo 🧪 Running quick validation tests...

REM Run quick tests
python run_tests.py quick
if %errorlevel% equ 0 (
    echo.
    echo 🎉 SETUP SUCCESSFUL!
    echo.
    echo The Brain-Computer Compression Toolkit is ready to use!
    echo.
    echo Next steps:
    echo   1. Activate the environment: venv\Scripts\activate.bat
    echo   2. Run standard tests: python tests/run_tests.py standard
    echo   3. Try the EMG demo: python examples/emg_demo.py
    echo   4. Run comprehensive tests: python tests/run_tests.py comprehensive
    echo.
    echo Available test commands:
    echo   python tests/run_tests.py quick         # 2 minutes - basic functionality
    echo   python tests/run_tests.py standard      # 10 minutes - recommended
    echo   python tests/run_tests.py comprehensive # 30 minutes - full validation
    echo.
) else (
    echo.
    echo ❌ SETUP VALIDATION FAILED
    echo.
    echo Some quick tests failed. This could indicate:
    echo   - Missing dependencies
    echo   - Installation issues
    echo   - System compatibility problems
    echo.
    echo Please check the error messages above and try:
    echo   1. Manual dependency installation: pip install numpy scipy matplotlib
    echo   2. Check Python version: python --version (need 3.8+)
    echo   3. Try individual tests: python tests/test_simple_validation.py
    echo.
)

pause
