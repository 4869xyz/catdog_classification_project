@echo off
cd /d "%~dp0"
where xelatex >nul 2>nul
if errorlevel 1 (
  echo XeLaTeX was not found. Install TeX Live or MiKTeX, or use Overleaf.
  exit /b 1
)
xelatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 exit /b 1
xelatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 exit /b 1
echo Report generated: main.pdf
