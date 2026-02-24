# Flux-on-GPU

基于 FLUX.1-schnell 模型的高性能 GPU 文本生图 API 服务，支持 LoRA 模型扩展。

## 功能特性

- 🚀 基于 FLUX.1-schnell 快速生成高质量图像
- 🎯 支持 LoRA 模型加载和权重调节
- ⚡ GPU 加速推理，优化生成速度
- 🔧 多进程架构，提升并发处理能力
- 📡 RESTful API 接口，易于集成

## 环境要求

- Python 3.8+
- CUDA 兼容的 GPU
- 8GB+ GPU 显存（推荐）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python start_server.py
```

服务将在 `http://0.0.0.0:8001` 启动。

### 3. 测试生成

```bash
python test.py
```

## API 使用

### 生成图像

**POST** `/generate`

#### 请求参数

```json
{
  "prompt": "描述文本",
  "negative_prompt": "负面提示词（可选）",
  "num_inference_steps": 4,
  "guidance_scale": 3.5,
  "width": 512,
  "height": 512,
  "seed": null,
  "lora_weights": [1.0],
  "use_loras": [true]
}
```

#### 响应格式

```json
{
  "image": "base64编码的图像数据",
  "generation_time": 2.5
}
```

## 配置说明

### 模型配置

- **基础模型**: `black-forest-labs/FLUX.1-schnell`
- **工作进程数**: 可在 `app_flux.py` 中调整 `NUM_WORKERS`
- **GPU 设备**: 通过 `CUDA_VISIBLE_DEVICES` 环境变量指定

### LoRA 模型

在 `app_flux.py` 中的 `LORA_PATHS` 列表添加您的 LoRA 模型路径：

```python
LORA_PATHS = [
    "your_lora_model.safetensors",
    # 添加更多 LoRA 模型
]
```

## 部署建议

### AWS G6e 实例部署

推荐使用 AWS G6e 实例进行部署，具备以下优势：

- 高性能 NVIDIA GPU
- 优化的深度学习环境
- 弹性扩缩容能力

### 生产环境配置

1. 使用 Docker 容器化部署
2. 配置负载均衡器
3. 设置监控和日志
4. 启用 HTTPS

## 性能优化

- 调整 `num_inference_steps` 平衡质量与速度
- 根据 GPU 显存调整批处理大小
- 使用适当的图像分辨率


