import requests
import json
import base64
from PIL import Image
import io
import time

url = "http://18.215.62.104:8001/generate"

# 创建一个九宫格表情包，每个格子展示太乙真人的不同夸张表情
payload = {
    "prompt": "women's cartoon style, exaggerated expressions of Taiyi Zhenren, nine-panel grid",
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "width": 512,
    "height": 512,
}

start_time = time.time()
response = requests.post(url, json=payload)
result = response.json()
end_time = time.time()
print(end_time - start_time)

# 解码并显示图像
image_data = base64.b64decode(result["image"])
image = Image.open(io.BytesIO(image_data))
image.save("generated_image_3.png")
print("Image saved as generated_image.png")
