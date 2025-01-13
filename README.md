# Local RAG System Based on Qwen-local-rag

This is a local RAG system based on [Qwen-local-rag](
https://help.aliyun.com/zh/model-studio/use-cases/build-rag-application-based-on-local-retrieval?spm=a2c4g.11186623.help-menu-2400256.d_2_5.1dbc47bb11vtgs)

You can choose to use the local model or the cloud model. You can see the model path details in the `Config.py` file, and modify the `chat.py` and `create_kb.py` as you want.

# Quick start:

1. set up the environment
```
pip install -r requirements.txt
```
2. add the [api-key](https://bailian.console.aliyun.com/?apiKey=1#/api-key)
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

# Note:
1. The `File` folder is used to store the documents.
2. The `VectorStore` folder is used to store the vector store.
3. The `Config.py` file is used to store the model path.
4. The `chat.py` file is used to chat with the model, you can modify the prompt in this file.
5. The `create_kb.py` file is used to create the vector store.
