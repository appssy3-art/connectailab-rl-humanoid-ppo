@echo off
chcp 949 >nul
title [MuJoCo] Rolling Agent Simulator
cd /d "%~dp0"
echo ========================================================
echo   [MuJoCo] Rolling Ball Agent Simulator
echo ========================================================
python run.py
pause

