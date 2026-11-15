@echo off
chcp 65001 > nul
cd /d "%~dp0"
title LINK-01 PPO 강화학습 훈련기

cls
echo ========================================================
echo   [LINK-01] 2족 보행 로봇 PPO 강화학습 훈련기
echo ========================================================
echo.
echo   학습할 목표 스텝을 선택하세요:
echo.
echo     [1] 빠른 걷기 체득 (20만 걸음 - 약 2분)
echo     [2] 안정적인 보행 (50만 걸음 - 약 5분)
echo     [3] 마스터 보행 (100만 걸음 - 약 10분)
echo     [4] 초정밀 장거리 보행 (200만 걸음 - 약 20분)
echo     [Q] 종료
echo.
echo ========================================================
set /p sel="선택 (1~4): "

if "%sel%"=="1" set STEPS=200000
if "%sel%"=="2" set STEPS=500000
if "%sel%"=="3" set STEPS=1000000
if "%sel%"=="4" set STEPS=2000000
if /i "%sel%"=="q" exit

cls
echo --------------------------------------------------------
echo [LINK-01] %STEPS% 걸음 PPO 학습을 시작합니다...
echo (학습 중단 시 Ctrl+C 를 누르면 현재까지의 두뇌가 자동 저장됩니다)
echo --------------------------------------------------------
python link01-gym-ppo7/train_link01_ppo.py --steps %STEPS% --envs 8 --dir link01-gym-ppo7/my_walk
pause
