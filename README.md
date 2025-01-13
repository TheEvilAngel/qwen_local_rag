This is a local RAG system based on Qwen-local-rag.

You can choose to use the local model or the cloud model. You can see the model path details in the `Config.py` file, and modify the `chat.py` and `create_kb.py` as you want.

# Quick start:

1. set up the environment
```
pip install -r requirements.txt
```
2. add the api-key
```
# 用您的 DashScope API Key 代替 YOUR_DASHSCOPE_API_KEY
echo "export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'" >> ~/.bashrc

source ~/.bashrc

echo $DASHSCOPE_API_KEY
```
3. run the server
```
uvicorn main:app --port 7866
```
4. visit the web page `127.0.0.1:7866`