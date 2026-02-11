@echo off
:: UTF-8 для консоли
chcp 65001>nul

echo ========================================
echo Установка окружения Python 3.10 для Kinect 360
echo ========================================
echo.

set PYTHON_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe

if not exist "%PYTHON_PATH%" (
    echo Python не найден по пути: %PYTHON_PATH%
    pause
    exit /b
)

"%PYTHON_PATH%" -m venv kinect_env
call kinect_env\Scripts\activate.bat

:: Обновляем pip
python -m pip install --upgrade pip

:: Ставим библиотеки
pip install numpy opencv-python moderngl glfw imgui[glfw] pythonnet PyOpenGL PyOpenGL-accelerate

echo ========================================
echo Установка завершена!
echo Перед запуском main.py установите Kinect SDK 1.8
echo https://www.microsoft.com/en-us/download/details.aspx?id=40278
echo ========================================
echo.
pause
