@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run setup_and_run.bat to install dependencies.
    pause
    exit /b 1
)
if not exist "Data2.xlsx" (
    echo Copy the original Data2.xlsx to this project directory.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" train_export.py --excel Data2.xlsx --sheet "标准化数据"
if errorlevel 1 (
    echo Training/export did not finish successfully. Read the error output above.
    pause
    exit /b 1
)
echo Full-data Stacking export finished. Next double-click start_local.bat.
pause
