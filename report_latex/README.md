# LaTeX 课程报告

内容依据《作业一.docx》要求、模型实现以及 outputs_regularized 的完整实验记录编写。

## 使用方式

- 将整个文件夹上传到 Overleaf，主文件选择 main.tex，编译器选择 XeLaTeX。
- 或在安装了 TeX Live / MiKTeX（含 ctex、Fandol 字体等依赖）的电脑上，在本目录打开 CMD，执行 compile.bat。
- main.tex 开头的 studentname、studentid、studentclass 用于填写姓名、学号和班级；当前保留空白，没有虚构个人信息。
- figures 中的三张图直接复制自本次正则化实验，可随源码独立使用。

## 编译命令（CMD）

```bat
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

连续编译两次以生成目录和交叉引用，输出为 main.pdf。

当前工作环境未检测到 XeLaTeX，因此已检查源码结构、图表引用和结果一致性，尚未实际编译或检查 PDF 分页。本文未运行新的训练实验，也未更改模型或原始输出。
