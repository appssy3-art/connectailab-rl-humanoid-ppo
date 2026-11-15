@echo off
chcp 65001 > nul
cd /d "%~dp0"
title LINK-01 2족 보행 로봇 3D 관찰기

:MENU
cls
echo ========================================================
echo   [LINK-01] 2족 보행 피지컬 AI 로봇 3D 실시간 관찰기
echo ========================================================
echo.
echo   실행할 모드를 선택하세요:
echo.
echo     [1] 내가 학습한 PPO 두뇌 (20만 걸음 모델)
echo     [2] CPG 기준 보행 패턴 (Reference Gait - 완벽한 보행)
echo     [3] 학습 전 기본 서있기 (Zero Action)
echo     [4] 학습 전 무작위 행동 (Random Action - 뇌진탕 낙하)
echo     [Q] 종료
echo.
echo ========================================================
set /p sel="번호를 입력하세요 (1~4): "

if "%sel%"=="1" goto RUN_MODEL
if "%sel%"=="2" goto RUN_CPG
if "%sel%"=="3" goto RUN_ZERO
if "%sel%"=="4" goto RUN_RANDOM
if /i "%sel%"=="q" exit
goto MENU

:RUN_MODEL
cls
python view_link01_model.py --mode model
pause
goto MENU

:RUN_CPG
cls
python view_link01_model.py --mode cpg
pause
goto MENU

:RUN_ZERO
cls
python view_link01_model.py --mode zero
pause
goto MENU

:RUN_RANDOM
cls
python view_link01_model.py --mode random
pause
goto MENU
