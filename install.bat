@echo off
:: Set console to UTF-8
chcp 65001>nul

echo ========================================
echo Setting up Python 3.10 environment for KinectForge
echo ========================================
echo.

set PYTHON_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe

if not exist "%PYTHON_PATH%" (
    echo Python not found at: %PYTHON_PATH%
    pause
    exit /b
)

"%PYTHON_PATH%" -m venv kinect_env
call kinect_env\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install libraries
pip install numpy opencv-python moderngl glfw imgui[glfw] pythonnet PyOpenGL PyOpenGL-accelerate

echo ========================================
echo Setup completed!
echo Before running main.py, install Kinect SDK 1.8
echo https://www.microsoft.com/en-us/download/details.aspx?id=40278
echo ========================================
echo.
pause
