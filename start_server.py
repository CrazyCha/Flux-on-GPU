import torch.multiprocessing as mp
import uvicorn
import asyncio
from app_flux import app, start_workers, NUM_WORKERS

# 全局引用到这些变量，以便路由函数可以访问
import app_flux as app_module

if __name__ == "__main__":
    # 确保主进程中使用 spawn 方法
    # 这一行可能不必要，因为我们已经在 app.py 中设置了
    # 但为了安全起见，再次设置
    mp.set_start_method('spawn', force=True)
    
    # 创建共享队列和字典
    task_queue = mp.Queue()
    result_dict = mp.Manager().dict()
    
    # 在全局模块中设置这些变量，以便 FastAPI 路由可以访问
    app_module.task_queue = task_queue
    app_module.result_dict = result_dict
    
    # 启动工作进程
    print(f"Starting {NUM_WORKERS} workers...")
    worker_processes = start_workers(task_queue, result_dict)
    
    # 启动 FastAPI 服务器
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
