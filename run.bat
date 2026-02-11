@echo off
:: UTF-8 for console
chcp 65001>nul

echo ========================================
echo Running KinectPyEffects using the virtual environment
echo ========================================
echo.

if not exist "kinect_env\Scripts\activate.bat" (
    echo Virtual environment not found! Please run install.bat first.
    pause
    exit /b
)

call kinect_env\Scripts\activate.bat
python main.py
pause