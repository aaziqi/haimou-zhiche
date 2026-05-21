# ICIP/ICASSP LaTeX Draft (MP-CycleGAN)

本文件夹包含按 **ICIP 5+1 页限制**整理的 LaTeX 初稿与配套图表（用于 IEEE SPS 的 `spconf` 模板）。

## 文件说明

- `paper.tex`：论文正文（通过宏控制是否匿名）。
- `paper_blind.tex`：匿名版入口（用于双盲投稿）。
- `paper_camera_ready.tex`：非匿名版入口（录用后camera-ready）。
- `refs.bib`：参考文献 BibTeX。
- `spconf.sty`：ICASSP/ICIP 常用样式文件。
- `figures/`：论文插图（来自你项目 `docs/figures`）。

## 编译（推荐顺序）

在本目录下运行：

```bash
pdflatex paper_blind.tex
bibtex paper_blind
pdflatex paper_blind.tex
pdflatex paper_blind.tex
```

camera-ready 版本同理，把入口文件换成 `paper_camera_ready.tex`。

> 备注：不同机器的 LaTeX 发行版可能已自带 `IEEEtran.bst`。本工程使用 `\bibliographystyle{IEEEtran}`。

