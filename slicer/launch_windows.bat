@echo off
rem Opens 3D Slicer with the TAVR Decide module loaded (Modules > Cardiac > TAVR Decide).
rem First time only, install the twin into Slicer's Python:
rem   "%LOCALAPPDATA%\slicer.org\3D Slicer 5.12.4\bin\PythonSlicer.exe" -m pip install -e "%~dp0.."
set "SLICER=%LOCALAPPDATA%\slicer.org\3D Slicer 5.12.4\Slicer.exe"
if not exist "%SLICER%" (
  echo 3D Slicer 5.12.4 not found. Install it from https://download.slicer.org
  pause
  exit /b 1
)
start "" "%SLICER%" --additional-module-paths "%~dp0TAVRDecide" --python-code "slicer.util.selectModule('TAVRDecide')"
