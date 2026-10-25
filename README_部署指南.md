# 🧬 微塑料 CCC Atlas｜Streamlit 部署操作指南

**当前 READY 项目已包含使用真实 `Data2.xlsx` 完整训练并验证导出的 `models/ccc_stacking.joblib`（约 1.8 MB），因此直接运行网站无需重新训练！** 最简单的流程请先打开 [`START_HERE_先看这个.md`](START_HERE_先看这个.md)。

> 本项目基于你上传的 **`Stacking_GBR750Enhanced_GroupSafe_All6`** 原始训练程序改造。
> 不是 CatBoost 网站，也不使用假数据预测。输入为训练日志中的 **13 个原始变量（7 数值 + 6 类别）**。
> 原来的程序**只打印控制台结果，不保存模型**，所以新增了 `train_export.py` 完成“训练全数据模型 → 保存六模型及五层元学习状态 → 离线验证 → 发布”。

## 一、先看文件结构

```text
microplastic_ccc_atlas/
├── app.py                       # 全新 Streamlit 网站：首页、表单、结果、导出
├── predictor.py                 # 逐层还原已拟合 Stacking 的在线推理
├── stacking_core.py             # 你的 4138 行训练算法 + 最小的导出状态扩展
├── train_export.py              # 用 Data2.xlsx 训练并保存最终部署模型
├── requirements.txt             # 启动及训练依赖；与部署保持同版本
├── setup_and_run.bat           # Windows 一键安装依赖并启动网站
├── train_local.bat              # 用原始 Data2.xlsx 重新训练最终模型
├── start_local.bat              # Windows 直接启动已安装的环境
├── smoke_test.py                # 完成模型导出后做一次预测可调用性自检
├── .gitignore                   # 训练 Excel 文件默认不上传 GitHub
├── .streamlit/
│   └── config.toml              # 网站主题
├── models/
│   ├── .gitkeep
│   ├── ccc_stacking.joblib      # 已附带：真实 507 条数据训练的完整参数 Stacking
│   └── model_info.json         # 已附带：本次完整训练的可核查元数据
├── README_部署指南.md
└── PAPER_METHODS.md
```

**这次已核对你新上传的真实 `Data2.xlsx`（507 条，45 个严格组），并完成完整参数的训练和预测一致性检查。压缩包包含真实模型，不包含原始 Excel，以降低误上传 GitHub 的风险。以后只有重新训练才需要把 `Data2.xlsx` 放到本地项目目录。**

## 二、Windows 电脑本地配置（推荐）

### 1. 解压项目，并放入训练数据

把项目文件夹解压到例如：

```text
D:\MicroplasticCCC\microplastic_ccc_atlas\
```

**READY 包中已带训练好的 `.joblib`；本地启动可跳过训练数据准备！** 如果你将来希望重训，才把论文原始 `Data2.xlsx` 放在这里；工作表名为 `标准化数据`，**不要用旧版 CatBoost 的 Data.xlsx**。

打开此文件夹，在资源管理器地址栏键入 `powershell` 并回车，进入这个文件夹的 PowerShell。之后的命令都在这里运行。

### 2. 安装 Python（建议 3.11）

电脑已有 Python 时，在 PowerShell 输入：

```powershell
py -3.13 --version
```

如果未安装 Python 3.13，可先从 https://www.python.org/downloads/ 安装。安装时确保已加入 PATH（Windows 安装程序可选择）。

建立隔离环境：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

此项目为了重现论文模型，约束了科学计算库版本。**训练时和云端部署时使用相同的 `requirements.txt` 和 Python 3.13**，可以减少 joblib 反序列化和版本差异问题。

> 如果系统只有一个合适的 Python 解释器，也可以把 `py -3.13` 替换成 `python`，但建议确认 `python --version`。

### 3. 用你的真实 Excel 训练并导出 Stacking 模型（仅首次及换数据时操作）

```powershell
.\.venv\Scripts\python.exe train_export.py --excel Data2.xlsx --sheet 标准化数据
```

这个命令会：

1. 原样读取 13 列 X、CCC 目标和文献编号；
2. 按原程序生成严格组，选取 KNN/Median 预处理；
3. 分别生成 RF / ExtraTrees / LGBM / XGB / GBR150 / GBR750 的训练侧严格 OOF 预测；
4. 在完整有效数据上拟合 6 个基础模型（两种 GBR 使用原双随机种子平均）；
5. 按原代码执行五层 Stacking、重复 Group Meta-CV 和 Champion Guard；
6. 将拟合好的预处理器、6 个基础模型、Stacking 各层已学习状态保存；
7. **计算同一批核对输入两条推理路径，要求二者数值一致，否则拒绝导出**。

成功后出现：

```text
models/ccc_stacking.joblib
models/model_info.json
```

文件 `ccc_stacking.joblib` 才是网站真正使用的模型，它不是运行日志，也不是你旧的 `model_bundle.pkl`。

**训练脚本不运行外层五折测试**：它为已经通过论文验证的模型构建一个供网站使用的**全数据最终拟合版本**。你的论文性能指标仍应引自原来的严格外层交叉验证日志，而不是把最终拟合时的训练侧分数当成测试成绩。

### 4. 在本地启动网站

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器通常会自动打开：

```text
http://localhost:8501
```

如果浏览器没有自动打开，手动复制这个地址即可。

网页中要**逐个**选择 6 个类别变量、键入 7 个数值变量，点击 `CALCULATE CCC`。右侧将显示 CCC（mM）、log10(CCC)、训练范围提示；还可以导出记录输入及预测值的 CSV。

- 页面缺模型：先检查 `models/ccc_stacking.joblib` 是否存在。
- 分类下拉框空：检查训练 Excel 的实际类别是否非空。
- 找不到工作表：运行 `--sheet` 指定真实表名。
- 不要把 CCC 目标列当成 X，也不要拿 old CatBoost 模型文件来替换。

Windows 还可以在完成上述准备步骤后**直接双击 `start_local.bat`** 启动。你也可以先运行 `python smoke_test.py` 检查真实导出的模型是否能调用（程序里的自动生成条件**仅用于技术自检，不代表实验结果**）。

停止本地网站：回到 PowerShell 窗口按 **Ctrl+C**。

### macOS / Linux 的命令

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python train_export.py --excel Data2.xlsx --sheet 标准化数据
./.venv/bin/python -m streamlit run app.py
```

## 三、将网站上传到你的 GitHub

**READY 包已经包含真实的 `models/ccc_stacking.joblib`，不需要先重新训练。**

1. 打开 https://github.com/ 并登录已有 GitHub 账号。
2. 右上角 `+` → **New repository**。
3. `Repository name` 输入 `microplastic-ccc-atlas`。
4. 论文要对外开放时建议 `Public`（公开）。不要在创建时勾选自动生成 README（已有项目文件）。
5. 点击 **Create repository**。
6. 进入仓库，找到 **uploading an existing file** 或 `Add file → Upload files`，上传解压后的**项目文件内容**而不是 ZIP（需要保留根目录下的 `app.py`、`requirements.txt`、`predictor.py`、`stacking_core.py`、`train_export.py`、`models` 和 `.streamlit` 的目录结构）。
7. **务必包含** `models/ccc_stacking.joblib`，以及 `.streamlit/config.toml`。如果 GitHub 网页上传没有保留隐藏目录，可以先跳过主题文件，其他文件可照常运行；后续用 Git 上传。
8. 在仓库网页确认：根目录直接能看到 `app.py` 和 `requirements.txt`，点击 `models/` 可以看到 `ccc_stacking.joblib`，然后 Commit 上传。

### 更稳妥：用 Git 命令上传（文件夹结构不会出错）

如果本地安装了 Git，打开项目目录 PowerShell：

```powershell
git init
git add .
git commit -m "Deploy microplastic six-learner CCC stacking atlas"
git branch -M main
git remote add origin https://github.com/你的GitHub用户名/microplastic-ccc-atlas.git
git push -u origin main
```

仓库 URL 里的 `你的GitHub用户名` 必须替换成你自己的 GitHub 登录名。如果远端仓库有预置提交而提示冲突，请先同步仓库或改用新空仓库。

**不要上传 `Data2.xlsx`**、原文文献表或其他需要保密的数据。此项目的 `.gitignore` 默认阻止训练 Excel 进入 Git，实际上传前可用 `git status` 检查。保存的 sklearn/GBR/XGB 模型可能保留训练统计特征；如果数据协议不允许公开模型本身，请使用私人仓库并确认分享策略。

如果模型文件超过 GitHub 单文件常规大小限制，**不要**用 GitHub 网页强行上传；优先检查模型文件体积并按 GitHub / Streamlit 官方指南设置 Git LFS，或改用支持模型托管的平台。Streamlit 文档说明其社区云支持从使用 LFS 的 GitHub 仓库读取文件。

## 四、免费部署 Streamlit Community Cloud

1. 打开 **https://share.streamlit.io/**。
2. 选择 **Continue with GitHub**，按提示授权你刚才使用的 GitHub 账户。
3. 进入工作区点击右上角 **Create app**。
4. 选择 **Yup, I have an app**（已有 GitHub 项目）。
5. 填：
   - **Repository**：`你的GitHub用户名/microplastic-ccc-atlas`
   - **Branch**：`main`
   - **Main file path**：`app.py`
   - **App URL**：自选，例如 `microplastic-ccc-atlas`（实际名称看可用性）。
6. 点 **Advanced settings**，Python 版本选择 **3.11**，保存设置。
7. 点击 **Deploy**，系统会依据 `requirements.txt` 安装依赖，加载 GitHub 中的已训练模型。
8. 完成后获得形如 `https://microplastic-ccc-atlas.streamlit.app/` 的公开链接（域名仅是示例，以平台实际分配为准）。把**真实的在线链接**放在论文正文、附录或二维码中。

**这里绝对不要在 Streamlit Cloud 里跑 `train_export.py`**：它应当在你自己的电脑上使用私有训练 Excel 训练一次。云端只负责读取模型并即时推理。

官方操作文档：

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization

## 五、上线后怎么更新

- **只改网页的文字、颜色、布局**：修改 `app.py` / `.streamlit/config.toml` 并上传 GitHub，Community Cloud 会自动读取新版本。
- **更新模型或 Excel 内容**：本地重新运行 `train_export.py` → 替换上传 `models/ccc_stacking.joblib` 及 `models/model_info.json` → GitHub 提交 → Community Cloud 自动重载。
- **修改 Python / sklearn / XGBoost 版本**：两边保持同一套依赖并重新导出模型；Cloud 的 Python 主版本如果需要更换，按官方文档重新部署。
- 出错时在 Streamlit Community Cloud 的 **Manage app → Logs** 查看详细报错。

## 六、科研与论文注意事项

1. **模型真实对应**：完整的六基础模型加五层 Stacking，不是 `VotingRegressor` 或一个新建的简化元模型。若任意层的 Meta-CV 认为修正不安全，网站会使用**训练时实际选中的回退分支**。
2. **指标口径**：你提供的原始控制台报告显示：
   - 样本：507 个有效样本，45 篇文献 / 45 个严格组。
   - 论文汇总主表 Stacking **pooled outer-OOF R²(log10 CCC) = 0.8235**，RMSE = 0.3043，MAE = 0.2415。
   - 平均各外层折 R²(log10 CCC) = **0.8136 ± 0.0642**（标准差）。
   - 原尺度 pooled outer-OOF R²(CCC) = **0.5113**，不要误把 log10 指标写成 CCC 原尺度指标。
3. **输入域**：训练中出现过的类别由模型包自动提取；数值使用训练范围提示，但不会把外推输入偷偷截断。范围是*逐变量*检查，不代表联合适用域保证。
4. **输出单位**：原代码对 `log10(CCC)` 训练，最终计算 `CCC = 10 ** predicted_log10_CCC`，单位是 **mM**。
5. **可重复性**：`train_export.py` 仅添加部署所需的全数据最终拟合流程，不会自动重做你论文中的全部五折成绩。发表前宜对导出的模型和论文所述算法口径重新人工确认。
6. **项目完整性**：不要把日志、占位文件、旧 CatBoost 模型伪装成真实部署权重。上线后用几组已知实验条件独立检查预测合理性。

## 七、常见故障快速排查

| 现象 | 处理办法 |
|---|---|
| `FileNotFoundError: Data2.xlsx` | 把真正训练用的 Excel 放入项目根目录，确认扩展名正确 |
| `Worksheet named '标准化数据' not found` | 核实 Excel 标签页名称，调整 `--sheet` |
| `ModuleNotFoundError` | 确认进入正确目录并在当前虚拟环境运行 `pip install -r requirements.txt` |
| `找不到已训练模型文件` | 先本地运行 `train_export.py`，再启动 Streamlit |
| 类别/输入列不匹配 | 核对真实 Excel 的列名，不能用旧版 Data.xlsx |
| GitHub 页面上传找不到子目录 | 使用 Git 命令上传，保留 `.streamlit/` 与 `models/` |
| Cloud 启动时报 sklearn 版本问题 | Python 和 requirements 保持与训练时相同，必要时重训导出 |
| Cloud 启动内存不足 | 检查模型包大小和部署资源；可换更充足的托管环境 |
| CCC 极小或极大 | 检查温度、浓度、粒径单位和外推范围；本项目输出并未宣称是实验真值 |

**一句话流程：解压 READY 包 → 双击 `setup_and_run.bat` → 浏览器本地测试 → GitHub 上传代码 + `.joblib`（不传 Excel）→ share.streamlit.io 选择 Python 3.13 部署 → 拿到公开网址后放入论文。**
