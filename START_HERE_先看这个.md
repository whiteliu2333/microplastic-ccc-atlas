# 微塑料 CCC Atlas：从本地到论文在线链接

## 先确认模型是否已包含

**本次 READY 版压缩包已包含 `models/ccc_stacking.joblib`（真实数据训练，完整参数，非演示模型）。直接按第一部分运行即可。** 如果以后删除或替换了权重文件，再看第二部分重新训练。

## 第一部分：直接用已经训练好的模型启动网站（Windows）

**1.** 将完整项目压缩包解压到电脑上，例如 `D:\Microplastic-CCC\microplastic_ccc_atlas`。**不要直接双击 ZIP 内的 app.py。**

**2.** 若没有 Python 3.13，从 https://www.python.org/downloads/ 安装 Python 3.13（安装时勾选 Add Python to PATH）。推荐使用**同一主版本 3.13**训练、运行、云端部署。

**3.** 在解压出来的项目文件夹中，**双击 `setup_and_run.bat`**，它会创建 `.venv`、安装依赖、启动 Streamlit。第一次安装取决于网络情况。

**4.** 浏览器打开 `http://localhost:8501`。页面上手动选择 6 个类别、填写 7 个数值，点击 **CALCULATE CCC**，得到 `CCC (mM)`；可点击 **Export result** 导出含输入条件的 CSV。

**5.** 以后启动只需**双击 `start_local.bat`**，不需要每次重新训练模型。关闭时在启动命令窗口按 `Ctrl+C`。

### 如果不想双击，也可使用 PowerShell

在项目文件夹的资源管理器地址栏输入 `powershell` 回车：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 第二部分：想用原始 Excel 重新训练一次（可选）

把 `Data2.xlsx` 放到**项目根目录**（与 `app.py` 同级），确保工作表名为 **`标准化数据`**，运行：

```powershell
.\.venv\Scripts\python.exe train_export.py --excel Data2.xlsx --sheet 标准化数据
```

重新训练完成会覆盖/生成 `models/ccc_stacking.joblib` 和 `models/model_info.json`。程序运行的是原始 6 个基础回归模型 + 5 层 Stacking + Champion Guard，**不是精简模型**；训练时会执行重复 Group Meta-CV，运行时间可能较长。

训练成功可检查模型能否正常预测：

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

`smoke_test.py` 中的组合仅用于软件可调用性检查，**不是论文实验点，也不是验证精度**。

## 第三部分：将网站发到 GitHub

**1.** 登录 https://github.com → 右上角 `+` → **New repository** → 名称填写 `microplastic-ccc-atlas` → **Public** → **Create repository**。

**2.** 上传解压后的项目**内容**到仓库根目录（建议先按下述 Git 命令上传，这样 `.streamlit` 隐藏文件夹也不会漏掉）。

**绝对不要把 `Data2.xlsx` 上传到公开仓库。** Git 忽略规则已设置；**必须上传** `models/ccc_stacking.joblib` 和根目录 `requirements.txt`，否则网站无法预测。

Windows PowerShell（须先安装 Git）：

```powershell
git init
git add .
git commit -m "Deploy microplastic CCC stacking atlas"
git branch -M main
git remote add origin https://github.com/你的用户名/microplastic-ccc-atlas.git
git push -u origin main
```

用自己的 GitHub 用户名替换 `你的用户名`。如果 Git 提示未设置身份，先执行 `git config --global user.name "你的昵称"` 以及 `git config --global user.email "你的邮箱"`，然后再次执行 commit。

**上传后在 GitHub 网页检查**：根目录有 `app.py`， `requirements.txt`，`.streamlit/config.toml`；`models/` 目录里有 `ccc_stacking.joblib`。

## 第四部分：部署 Streamlit Cloud（无需 GitHub 插件）

**1.** 打开 https://share.streamlit.io/ ，使用 GitHub 账号登录。

**2.** 点击 **Create app** → **Yup, I have an app**。

**3.** 填入：

- **Repository:** `你的用户名/microplastic-ccc-atlas`
- **Branch:** `main`
- **Main file path:** `app.py`
- **App URL:** 自选可用域名

**4.** 点击 **Advanced settings**，在 **Python version** 里选择 **3.13**（与导出模型环境一致），保存。

**5.** 点击 **Deploy**。平台安装依赖和加载模型后，会给出 `https://你的域名.streamlit.app/` 形式的网址。先亲自用一组真实实验输入测试，再将正式网址写入论文或制作二维码。

重要：代码和模型通过 GitHub 上传；**不需要，也不应该把训练 Excel 上传到 Streamlit Cloud**。

官方文档：
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization

## 第五部分：科研严谨性

- 旧日志汇总的严格外层五折 OOF `R²_log10(CCC) = 0.8235` 是**论文原验证结果**，不是部署模型的全数据训练分数。
- 部署版本是用 507 个样本再次训练生成的最终拟合模型，不会通过重新计算训练预测来冒充独立测试结果。
- CCC 输出使用 mM。模型先在 `log10(CCC)` 尺度预测，再用 `10 ** 预测值` 还原。
- 自变量超出训练范围时网站给出提示；不代表模型在范围外必然可靠。
- 原始数据中 `光照时间h X11` 有 98/507 条有效记录，余下记录的缺失由训练时预处理策略处理；**网站表单仍要求用户明确输入实验值**。

## 最常见的错误

- **页面提示 MODEL NOT INSTALLED**：没有在 `models/` 目录放置 `.joblib`。运行第二部分训练，或使用包含模型的完整项目包。
- **Python command not found**：电脑没装 Python 3.13，或没有正确配置 PATH。
- **找不到标准化数据工作表**：检查 Excel 底部的工作表标签。
- **Cloud 依赖安装失败**：核对 Python 3.13、`requirements.txt`、Cloud 运行日志。
- **GitHub 提示文件太大**：如果单个 `.joblib` 超过 GitHub 100MB 限制，需要使用 Git LFS；详见 GitHub/Streamlit 官方文档。
