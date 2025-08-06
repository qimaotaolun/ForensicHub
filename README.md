# Baseline

本项目是在 [ForensicHub](https://github.com/scu-zjz/ForensicHub) 基础上开发的篡改检测任务Baseline模型，采用 ConvNeXt-Large 作为backbone架构，专注于全篡改场景下的图像篡改检测。模型代码位于 `tasks/bisai` 目录，训练配置文件为 `statics/bisai/bisai_train.yaml`。

## 环境安装

请按照以下步骤设置运行环境：

1. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

2. 安装本地包（开发模式）：

   ```bash
   pip install -e .
   ```

## 训练方式

执行以下脚本开始训练：

```bash
sh statics/run.sh
```

## 推理方式

执行以下脚本开始推理：

```bash
python training_scripts/inference_bisai.py
```

## 项目结构

* `statics/`：包含运行脚本和其他静态资源；
* `tasks/bisai`：主代码目录；
* `requirements.txt`：依赖文件；
* `setup.py`：安装配置文件。

## 参考

本项目基于 [ForensicHub](https://github.com/scu-zjz/ForensicHub) 进行二次开发，感谢其开源贡献。
