@echo off
echo Installing required packages for disaster monitoring...
echo.

REM Try to install packages
pip install feedparser
if %errorlevel% neq 0 (
    echo Failed to install feedparser
    goto :error
)

pip install beautifulsoup4
if %errorlevel% neq 0 (
    echo Failed to install beautifulsoup4
    goto :error
)

pip install lxml
if %errorlevel% neq 0 (
    echo Failed to install lxml
    goto :error
)

echo.
echo All packages installed successfully!
echo You can now run: python disaster_monitor.py
echo.
pause
goto :end

:error
echo.
echo Installation failed. Please try running this script as administrator
echo or install packages manually:
echo   pip install feedparser beautifulsoup4 lxml
echo.
pause

:end