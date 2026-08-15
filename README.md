# Power RAG Assistant

一个面向电力知识场景的轻量级 `RAG` 智能问答系统。项目支持导入电力巡检、安全规程、故障处理等文档，构建本地知识库，并通过检索增强生成回答，返回答案、参考资料和相似度。

## 项目特点

- 支持 `txt`、`pdf`、`docx` 多格式知识文档导入
- 基于本地 `TF-IDF + cosine similarity` 实现中文检索
- 支持接入 `OpenAI` 兼容接口，如 `DeepSeek`
- 提供 `FastAPI` 后端接口和 `Streamlit` 前端页面
- 支持显示参考片段和相似度，结果可解释
- 支持本地运行和 Docker 部署

## 项目结构

```text
power_rag_assistant/
├── app/
│   ├── api/                 # 问答接口
│   ├── core/                # 配置管理
│   ├── rag/                 # 文档加载、切块、检索
│   └── main.py              # FastAPI 入口
├── data/
│   ├── raw/                 # 原始知识文档
│   └── processed/           # 清洗后的文本
├── scripts/
│   └── build_index.py       # 构建索引脚本
├── vector_store/            # 本地检索索引
├── frontend.py              # Streamlit 前端
├── requirements.txt         # Python 依赖
├── Dockerfile               # Docker 镜像定义
└── docker-compose.yml       # Docker Compose 配置
```

## 技术架构

### 前端

- `Streamlit`
- 对话式问答页面
- 参考资料展开显示
- 参数配置，如 `Top K`、是否启用 LLM

### 后端

- `FastAPI`
- `Pydantic`
- `Uvicorn`
- 提供健康检查、问答接口、接口文档

### RAG 链路

1. 从 `data/raw` 读取原始文档
2. 解析 `txt`、`pdf`、`docx`
3. 对长文本进行切块
4. 构建本地检索索引
5. 用户提问时检索最相关片段
6. 将片段作为上下文发送给大模型
7. 返回答案和参考资料

## 环境要求

- Python `3.10` 及以上
- 推荐使用虚拟环境
- 如果启用大模型，需要可访问对应模型服务

## 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 环境变量配置

在项目根目录创建 `.env` 文件，示例：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
EMBEDDING_BACKEND=auto
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

说明：

- `OPENAI_API_KEY`：大模型 API Key
- `OPENAI_BASE_URL`：OpenAI 兼容接口地址
- `MODEL_NAME`：聊天模型名称
- `TOP_K`：默认召回片段数
- `CHUNK_SIZE`：切块长度
- `CHUNK_OVERLAP`：切块重叠长度

## 准备知识库

将你的知识文档放到 `data/raw` 目录下，支持格式：

- `.txt`
- `.pdf`
- `.docx`

例如：

- `power_inspection_guide.txt`
- `power_safety_guide.txt`
- `电力安全手册.docx`

## 构建索引

当你新增或修改了 `data/raw` 中的文档后，需要重新构建索引：

```powershell
.venv\Scripts\python.exe scripts\build_index.py
```

脚本会自动完成：

- 文档解析
- 文本清洗
- 文本切块
- 生成 `data/processed`
- 生成 `vector_store` 检索索引

## 本地启动

### 1. 启动后端

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后可访问：

- API 首页：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/health`
- Swagger 文档：`http://127.0.0.1:8000/docs`

### 2. 启动前端

新开一个终端执行：

```powershell
.venv\Scripts\streamlit.exe run frontend.py --server.address 0.0.0.0 --server.port 8501
```

启动后访问：

- 前端页面：`http://localhost:8501`

## Docker 启动

如果本机已安装 Docker 和 Docker Compose，可直接运行：

```powershell
docker compose up --build
```

启动后访问：

- 前端：`http://localhost:8501`
- 后端：`http://127.0.0.1:8000`

## API 使用示例

### 测试健康状态

```http
GET /health
```

返回：

```json
{
  "status": "ok"
}
```

### 问答接口

```http
POST /api/v1/chat
Content-Type: application/json
```

请求示例：

```json
{
  "query": "变压器巡检要点有哪些？",
  "top_k": 3,
  "use_llm": true
}
```

返回示例：

```json
{
  "answer": "根据提供的知识库内容，变压器巡检要点包括……",
  "sources": [
    {
      "chunk_id": "doc-0-chunk-0",
      "source": "data/raw/power_inspection_guide.txt",
      "text": "电力设备巡检与故障处理指南……",
      "score": "0.1610"
    }
  ]
}
```

## 检索实现说明

当前版本默认使用本地 `TF-IDF` 检索，适合轻量部署和中文场景：

- 向量器：字符级 `n-gram`
- 相似度：`cosine similarity`
- 索引文件：
  - `vector_store/manifest.json`
  - `vector_store/metadata.json`
  - `vector_store/tfidf_index.npz`
  - `vector_store/tfidf_vectorizer.pkl`

## 常见问题

### 1. 新增文档后为什么问不到？

因为知识库索引没有更新。请重新执行：

```powershell
.venv\Scripts\python.exe scripts\build_index.py
```

### 2. 大模型调用失败怎么办？

优先检查：

- `.env` 中的 `OPENAI_API_KEY`
- `.env` 中的 `OPENAI_BASE_URL`
- `.env` 中的 `MODEL_NAME`
- 本机网络是否可访问模型服务

### 3. 相似度一直很低怎么办？

可尝试：

- 增加更贴近业务的原始文档
- 调整 `CHUNK_SIZE` 与 `CHUNK_OVERLAP`
- 调整提问方式，尽量使用明确的设备名或业务场景

### 4. 端口被占用怎么办？

- 后端默认端口：`8000`
- 前端默认端口：`8501`

如果端口被占用，可停止原有进程或改用其他端口。

## 适用场景

- 电力巡检知识查询
- 安全规程辅助问答
- 故障处理经验检索
- 企业内部知识助手
- 智能运维知识服务原型

## 后续可扩展方向

- 替换为向量数据库，如 `Milvus`、`pgvector`
- 增加多用户权限与知识库管理
- 接入企业内部设备台账与工单系统
- 增加更细粒度的引用与日志分析能力

## 许可证

如需开源发布，请根据你的实际需求补充 License 文件。
