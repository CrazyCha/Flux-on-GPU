import os
import io
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import base64
from PIL import Image
from diffusers import FluxPipeline
import time
import asyncio
import uuid
from safetensors.torch import load_file
import random

# 确保在导入时就设置多进程启动方法
mp.set_start_method('spawn', force=True)
  
# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个 GPU
  
# 启动工作进程数
NUM_WORKERS = 1  # 4
  
app = FastAPI(title="FLUX.1-schnell Text-to-Image API with LoRA support")
  
# 模型配置
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# LoRA 模型路径列表
LORA_PATHS = [
    #"哪吒Flux模型_V2.0.safetensors",
    #"太乙真人flux版_v1.0.safetensors",
    #"表情包-万物皆有表情9_v1.0.safetensors"
]
  
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 3.5
    max_sequence_length: Optional[int] = 512
    seed: Optional[int] = None
    height: Optional[int] = 512
    width: Optional[int] = 512
    lora_weights: Optional[List[float]] = None  # 每个LoRA模型的权重
    use_loras: Optional[List[bool]] = None  # 选择使用哪些LoRA模型
  
class GenerationResponse(BaseModel):
    image: str  # base64 encoded image
    seed: int   # 返回使用的种子，便于复现
  
def worker_process(gpu_id, task_queue, result_dict):
    """工作进程函数，负责加载模型和处理请求"""
    try:
        print(f"Worker {gpu_id} starting up...")
        # 设置 CUDA 上下文
        torch.cuda.set_device(0)  # 所有进程使用同一个 GPU，但会分时复用

        # 加载基础模型
        pipeline = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16)
        
        # 加载所有LoRA模型
        loaded_loras = []
        for lora_path in LORA_PATHS:
            print(f"Loading LoRA model from {lora_path}")
            
            # 根据文件扩展名选择合适的加载方法
            if lora_path.endswith('.safetensors'):
                lora_state_dict = load_file(lora_path)
            else:
                lora_state_dict = torch.load(lora_path, map_location="cpu")
                
            loaded_loras.append(lora_state_dict)
        
        # 创建标记当前加载的适配器名称的集合
        active_adapters = set()
        
        pipeline = pipeline.to("cuda")
        print(f"Worker {gpu_id} initialized and ready to process requests")

        # 持续从队列获取任务并处理
        while True:
            try:
                task_id, request_dict = task_queue.get()
                print(f"Worker {gpu_id} processing task {task_id}: {request_dict['prompt'][:30]}...")
                
                # 应用LoRA权重
                if 'use_loras' in request_dict and request_dict['use_loras'] is not None:
                    use_loras = request_dict['use_loras']
                    lora_weights = request_dict.get('lora_weights', [1.0] * len(LORA_PATHS))
                    
                    # 确保权重列表长度和use_loras一致
                    if len(lora_weights) < len(use_loras):
                        lora_weights.extend([1.0] * (len(use_loras) - len(lora_weights)))
                    
                    # 为选定的LoRA模型应用权重
                    active_adapter_names = []
                    active_adapter_weights = []
                    
                    # 为每个请求生成唯一的请求ID前缀
                    request_uid = str(uuid.uuid4())[:8]
                    for i, (use_lora, weight) in enumerate(zip(use_loras, lora_weights)):
                        if use_lora and i < len(loaded_loras):
                            # 使用请求ID作为适配器名称的一部分，确保唯一性
                            adapter_name = f"lora_{request_uid}_{i}"
                            # 应用LoRA与权重
                            pipeline.load_lora_weights(loaded_loras[i], adapter_name=adapter_name, scale=weight)
                            active_adapter_names.append(adapter_name)
                            active_adapter_weights.append(weight)
                            active_adapters.add(adapter_name)
                    
                    print('active_adapter_names:', active_adapter_names)
                    print('active_adapter_weights:', active_adapter_weights)
                    # 如果有活跃的适配器，设置它们
                    if active_adapter_names:
                        pipeline.set_adapters(active_adapter_names, adapter_weights=active_adapter_weights)
                        pipeline.enable_lora()
                
                # 设置随机种子，如果没有提供则生成一个
                seed = request_dict.get('seed')
                if seed is None:
                    seed = random.randint(0, 2147483647)  # 生成随机种子
                
                # 应用种子
                generator = torch.Generator("cpu").manual_seed(seed)
                
                # 生成图像
                image = pipeline(
                    prompt=request_dict['prompt'],
                    negative_prompt=request_dict.get('negative_prompt'),
                    num_inference_steps=request_dict.get('num_inference_steps', 40),
                    guidance_scale=request_dict.get('guidance_scale', 3.5),
                    max_sequence_length=request_dict.get('max_sequence_length', 512),
                    height=request_dict.get('height', 512),
                    width=request_dict.get('width', 512),
                    generator=generator
                ).images[0]
                
                # 转换为 base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 存储结果，包含使用的种子
                result_dict[task_id] = {"image": img_str, "seed": seed}
                print(f"Worker {gpu_id} completed task {task_id}")
            except Exception as e:
                print(f"Worker {gpu_id} encountered error: {str(e)}")
                # 存储错误信息
                if 'task_id' in locals():
                    result_dict[task_id] = {"error": str(e)}
            finally:
                # 在处理完请求后，添加这段代码
                try:
                    # 更彻底地卸载所有LoRA适配器
                    if active_adapters:
                        pipeline.disable_lora()
                        # 尝试手动删除适配器
                        for adapter_name in active_adapters:
                            if hasattr(pipeline, "delete_adapter") and callable(getattr(pipeline, "delete_adapter")):
                                pipeline.delete_adapter(adapter_name)
                        active_adapters.clear()
                except Exception as e:
                    print(f"Error cleaning up adapters: {str(e)}")
    except Exception as e:
        print(f"Worker {gpu_id} initialization failed: {str(e)}")
  
# 直接初始化任务队列和结果字典
# 等主函数里初始化
task_queue = None
result_dict = None
  
@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    # 生成唯一的任务 ID
    task_id = str(uuid.uuid4())
    # 将请求放入队列
    task_queue.put((task_id, request.dict()))
    print(f"Added task {task_id} to queue")
    # 等待结果
    max_wait_time = 300  # 最大等待时间（秒）
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if task_id in result_dict:
            result = result_dict[task_id]
            # 删除结果以释放内存
            del result_dict[task_id]
            # 检查是否有错误
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        # 短暂睡眠，避免忙等待
        await asyncio.sleep(0.5)
    # 超时
    raise HTTPException(status_code=504, detail="Request timed out")
  
@app.get("/health")
async def health_check():
    return {"status": "healthy", "workers": NUM_WORKERS}
  
@app.post("/generate_view")
async def generate_image_view(request: GenerationRequest):
    result = await generate_image(request)
    image_bytes = base64.b64decode(result["image"])
    return Response(content=image_bytes, media_type="image/png")
  
def start_workers(task_queue, result_dict):
    """启动工作进程"""
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(i, task_queue, result_dict))
        p.daemon = True
        p.start()
        processes.append(p)
    return processes
