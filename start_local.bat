@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing local Python environment.
  echo Install Python 3.11 then run:
  echo py -3.11 -m venv .venv
  echo .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
if not exist "models\ccc_stacking.joblib" (
  echo [ERROR] Missing REAL trained Stacking artifact.
  echo Copy your original Data2.xlsx into this folder and run:
  echo .venv\Scripts\python.exe train_export.py --excel Data2.xlsx --sheet 标准化数据
  pause
  exit /b 1
)
echo Starting CCC ATLAS...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
