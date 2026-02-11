@echo off
:: UTF-8 для консоли
chcp 65001>nul

echo ========================================
echo Запуск main.py через виртуальное окружение
echo ========================================
echo.

if not exist "kinect_env\Scripts\activate.bat" (
    echo Виртуальное окружение не найдено! Сначала запустите install_kinect_env.bat
    pause
    exit /b
)

call kinect_env\Scripts\activate.bat
python main.py
pause