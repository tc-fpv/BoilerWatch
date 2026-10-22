@echo off
setlocal
cd /d "%~dp0"

if not defined LAST_IMAGE_PATH (
    findstr /r /c:"^[ ]*LAST_IMAGE_PATH[ ]*=" ".env" >nul 2>nul
    if errorlevel 1 (
        if defined RTSP_URL (
            set "LAST_IMAGE_PATH=%~dp0data\last.jpg"
        ) else (
            findstr /r /c:"^[ ]*RTSP_URL[ ]*=" ".env" >nul 2>nul
            if errorlevel 1 (
                set "LAST_IMAGE_PATH=%~dp0tests\boiler.PNG"
            ) else (
                set "LAST_IMAGE_PATH=%~dp0data\last.jpg"
            )
        )
    )
)

if not exist "%~dp0data" mkdir "%~dp0data"

py -3.12 -m boiler_reader.image_viewer
if errorlevel 1 (
    echo.
    echo Boiler Reader failed to start. Check Python 3.12 and install dependencies with:
    echo py -3.12 -m pip install -e .
    pause
)

endlocal