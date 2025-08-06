import os
# 升级方舟 SDK 到最新版本 pip install -U 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark

client = Ark(
    # 从环境变量中读取您的方舟API Key
    # api_key=os.environ.get("f40416fd-bfd1-4370-b74f-15c3338d0c58"),
    api_key="f40416fd-bfd1-4370-b74f-15c3338d0c58",
    # 深度思考模型耗费时间会较长，请您设置较大的超时时间，避免超时，推荐30分钟以上
    timeout=120,
    )
response = client.chat.completions.create(
    # 替换 <Model> 为模型的Model ID
    model="deepseek-r1-250528",
    # model="deepseek-r1-250120",
    messages=[
        {"role": "user", "content": "我要有研究推理模型与非推理模型区别的课题，怎么体现我的专业性"}
    ],
    # thinking={
    #     "type": "enable"
    # }
)
# 当触发深度思考时，打印思维链内容
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)
