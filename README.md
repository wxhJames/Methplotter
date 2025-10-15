# MethPlotter  
一个零代码、轻量级的 DNA 甲基化数据可视化 Python 工具包  
*A zero-coding, lightweight Python toolkit for DNA-methylation visualisation*

---

## 🧬 功能速览 | Features  
| 中文 | English |
|---|---|
| ✅ 读取 CX-report & GFF3 两种主流格式 | Read CX-report & GFF3 out-of-the-box |
| ✅ 一行命令绘制折线 / 热图 / 小提琴 / 柱状图 | One-line CLI for line-plot, heatmap, violin, bar |
| ✅ 自动缺失值填补、异常值过滤、染色体排序 | Auto missing-value fill, outlier removal, chrom-sort |
| ✅ 支持局部区域上下游扩展可视化 | Up-/down-stream flank visualisation |
| ✅ 发布级 PDF/PNG 双输出 | Publication-ready PDF & PNG simultaneously |
| ✅ Jupyter 模板，即插即用 | Jupyter notebook template included |

---

## 🚀 快速开始 | Quick Start  
### 1. 安装 Install
```bash
pip install MethPlotter
```

### 2. 1 分钟跑通 Example (中文)
```bash
# 下载测试数据（已内置）
methplotter --demo

# 绘制 5 号染色体整体甲基化趋势
methplotter --cx Tair10.CX_report.txt \
            --gff TAIR10.gff \
            --chr Chr5 \
            --plot line \
            --outdir my_figs
```
打开 `my_figs/Chr5_methylation_line.pdf` 即可。

### 2. Quick start (English)
```bash
# built-in demo dataset
methplotter --demo

# genome-wide line-plot for Chr5
methplotter --cx Tair10.CX_report.txt \
            --gff TAIR10.gff \
            --chr Chr5 \
            --plot line \
            --outdir my_figs
```
Check `my_figs/Chr5_methylation_line.pdf`.

---

## 📥 输入格式 | Input
| 文件 | 说明 | 必需 |
|---|---|---|
| `*.CX_report.txt` | 由 Bismark / BS-seeker 生成的甲基化率文件 | ✔ |
| `*.gff` / `*.gff3` | 基因组注释 | ✔ |

---

## 📊 可选图表 | Plot types
| 图表 | 中文一句话说明 | One-line description |
|---|---|---|
| `line` | 染色体水平甲基化走势 | Chromosome-wide trend |
| `lineExtra` | 基因/区域前后 N kb 细节 | Zoom-in ±N kb flanking region |
| `cluster` | 多区域聚类热图 | Cluster heat-map across regions |
| `violin` | 指定甲基化类型分布 | Distribution of CG/CHG/CHH |
| `bar` | 区域间均值比较 | Mean methylation comparison |

---

## ⚙️ 进阶参数 | Advanced CLI
```bash
methplotter --cx case.CX_report.txt ctrl.CX_report.txt \
            --labels Case Ctrl \
            --plot violin \
            --context CHG \
            --outdir diff_plots \
            --palette Set2 \
            --width 12 --height 6
```
Run `methplotter --help` for the full list.

---

## 🐍 Python API (可选)
```python
import MethPlotter as mp

cx  = mp.read_cx('Tair10.CX_report.txt')
gff = mp.read_gff('TAIR10.gff')

data = mp.generate_line_data(cx, chr='Chr5')
mp.draw_line(data, window=2e5, color=['#1f77b4'])
```
---

## 📦 依赖 | Dependencies
- Python ≥ 3.8
- pandas ≥ 2.2
- matplotlib ≥ 3.10
- seaborn ≥ 0.13
- numpy ≥ 2.1  
*所有依赖会自动随 pip 安装*  
*All deps are auto-installed via pip.*

---

## 🧪 测试数据 | Test data
```bash
methplotter --demo
```
命令会下载 **拟南芥** 1 号 & 5 号染色体示例数据 (~3 MB)。  
This pulls **Arabidopsis** Chr1 & Chr5 sample data (~3 MB).

---

## 📮 联系 | Contact
- GitHub Issues: [github.com/wxhJames/MethPlotter/issues](https://github.com/wxhJames/MethPlotter/issues)
- 邮箱 Email: weixin19375@163.com

---
