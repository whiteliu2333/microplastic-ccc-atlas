# Manuscript-ready description — Microplastic CCC Atlas

## English (adapt to your actual manuscript tense after deployment)

**Web-based prediction interface.** A research-oriented web application for the prediction of the critical coagulation concentration (CCC) of microplastics was implemented using Streamlit and deployed from a GitHub repository. The interface requires the user to specify all 13 experimental predictors, comprising seven numerical descriptors and six categorical descriptors. The numerical features cover experimental temperature, zeta potential, pH, plastic concentration, initial particle size, concentration of coexisting substances, and irradiation duration. The categorical descriptors cover cation valence, anion type, particle morphology, plastic surface modification, polymer type, and type of coexisting substance.

**Prediction engine.** The inference backend uses a six-base-model stacking regressor containing Random Forest, Extra Trees, LightGBM, XGBoost, a 150-estimator Gradient Boosting Regressor, and an internal 750-estimator Gradient Boosting Regressor used as a high-weight anchor. The stacking logic consists of constrained anchor-weighted blending, residual meta-calibration, sample-wise expert gating and scale correction, locally smoothed residual calibration, and a final stability-controlled shrinkage step. A champion-guard mechanism preserves the anchor prediction when its predefined training-side reliability checks are not satisfied. Numerical imputation, standardization and one-hot encoding follow the preprocessing configuration selected on the training side.

**Deployment and units.** A final prediction artifact is fitted to the available eligible training data, serialized together with its preprocessing transformation and learned meta-model parameters, and loaded by the Streamlit application for inference. The model estimates the base-10 logarithm of CCC, and the application converts the predicted value back to the original concentration scale in millimoles per liter (mM). Warnings are displayed for individual input values outside the ranges or categories observed during model development. Predictions are computational estimates and should not be interpreted as independent experimental measurements or calibrated uncertainty intervals.

**Cross-validation result (based on the supplied research run, not on the deployment artifact).** In the supplied validation log, the stacking framework achieved a pooled outer out-of-fold coefficient of determination of **R² = 0.8235** on the log10(CCC) scale, with RMSE = **0.3043** and MAE = **0.2415** on the same transformed scale. The corresponding pooled original-scale CCC R² was **0.5113**. These metrics must be attributed to the original outer group-separated five-fold experiment, not to fresh evaluations of the final full-data deployment fit.

> Editorial note: only use “deployed” and provide a public URL once you have actually deployed the GitHub repository and confirmed the live calculator works.

## 中文（可按论文风格调整）

为便于研究人员快速评估不同实验条件下微塑料的临界混凝浓度（CCC），本研究基于 Streamlit 构建了交互式在线预测平台。平台支持用户逐项输入 13 项变量，包括阳离子价态、阴离子类型、温度、Zeta 电位、颗粒形态、pH、塑料表面修饰类型、塑料种类、塑料浓度、初始粒径、共存物质种类、共存物质浓度及光照时间。后台集成六个一级回归模型（RF、ExtraTrees、LightGBM、XGBoost、GBR150 和作为内部锚点的 GBR750），在其基础上实施约束加权融合、残差校准、动态门控、分区校准及稳定收缩，并通过 Champion Guard 机制控制异常情况下的模型回退。模型在 log10(CCC) 尺度计算预测结果，平台再将其反变换为 mM 单位的 CCC 值。平台同时针对超出训练数据覆盖范围的单个输入变量给出外推提示，以增强输出解释的审慎性。

在用户提供的原始实验运行日志中，该集成模型的严格外层分组交叉验证 pooled OOF R²(log10 CCC) 为 0.8235。需要明确：该值来自论文原验证过程，不能替代部署模型的独立检验；部署版本是在全部有效训练数据上再次拟合所得。

## Figure caption (only if screenshots show the finished deployed interface)

**Figure X.** The CCC Atlas research web interface for predicting microplastic critical coagulation concentrations. The application accepts 13 user-specified experimental features, executes the pretrained six-learner, five-stage stacking predictor, and reports the estimated CCC in mM together with input-domain notices.

## Figure capture suggestion

Use a browser at 1440–1600 px width. Capture one image of the top dark-teal title section and input cards, and one of the completed experimental form alongside the populated CCC output panel. Replace sample values with realistic, published experimental conditions; do not fabricate a prediction result for a screenshot.
