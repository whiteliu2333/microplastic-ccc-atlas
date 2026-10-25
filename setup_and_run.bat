@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.13 --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.13 was not found.
        echo Download Python 3.13 from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo Could not create Python environment.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Review the output above.
    pause
    exit /b 1
)
if not exist "models\ccc_stacking.joblib" (
    echo No trained model found. Copy Data2.xlsx here and run:
    echo .venv\Scripts\python.exe train_export.py --excel Data2.xlsx --sheet "标准化数据"
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
