# CLAUDE.md

本文件用于指导 Claude Code 在本仓库中工作。请在所有对话中遵守本文件的约定。

## 项目名称

**AI 智能知识库 Agent 平台**

## 项目目标

开发一个功能完整、可直接写入简历的 **Agent 实习求职项目**。项目需要覆盖当前 AI 应用开发岗位最核心的三项能力：

- **Agent**：具备工具调用与多步推理能力的智能体
- **RAG**：基于私有知识库的检索增强生成
- **MCP**：通过 Model Context Protocol 接入外部工具与数据源

目标不是做一个「能跑的 demo」，而是做一个**结构清晰、能讲清楚、能部署上线**的完整项目，用于面试时展示工程能力。

## 技术栈

| 层次 | 技术选型 |
| --- | --- |
| 后端 | Python + FastAPI |
| 关系型数据库 | PostgreSQL |
| 向量数据库 | Milvus |
| 大语言模型 | DeepSeek |
| 前端 | Vue 3 |
| 部署 | Docker / Docker Compose |

## 规划中的目录结构

> 项目目前处于起步阶段，以下为目标结构。实际新增目录时请尽量与之保持一致。

```
.
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI 路由层
│   │   ├── core/         # 配置、日志、安全等基础设施
│   │   ├── models/       # 数据库模型（PostgreSQL）
│   │   ├── schemas/      # Pydantic 请求/响应模型
│   │   ├── services/     # 业务逻辑（RAG、Agent、MCP 客户端）
│   │   └── main.py       # FastAPI 应用入口
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # Vue 3 前端
│   ├── src/
│   └── Dockerfile
├── docker-compose.yml
├── .env.example          # 环境变量模板（可提交）
├── .env                  # 真实密钥（禁止提交）
└── CLAUDE.md
```

## 常用命令

> 命令会随项目推进逐步补充。注意本项目根目录路径含空格，命令行中需使用引号。

```bash
# 后端本地开发（示例）
cd backend && uvicorn app.main:app --reload

# 前端本地开发（示例）
cd frontend && npm run dev

# Docker 一键启动
docker compose up -d --build
```

## 代码规范

### 1. 密钥与配置（强制）

- **所有密钥、API Key、数据库密码必须通过环境变量读取，严禁硬编码在代码中。**
- 使用 `pydantic-settings` 统一从 `.env` 读取配置，代码中只引用配置对象，不直接调用 `os.environ`。
- 仓库中只保留 `.env.example`；`.env` 必须写入 `.gitignore`。
- 提交前自查：本次改动是否引入了任何明文密钥。

### 2. Python 代码风格

- **尽量为函数参数和返回值添加类型提示**（type hints）。
- 使用 Pydantic 模型定义接口的输入输出，避免裸 `dict` 在层与层之间传递。
- 使用异步写法（`async def`）处理 I/O 密集操作（数据库、HTTP 请求、向量检索）。
- 命名：变量/函数用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_CASE`。

### 3. 代码可读性

- **保持代码简单、清晰，适合初学者阅读和学习。**
- 优先选择直白的实现，而不是「聪明」的技巧。
- 关键逻辑（尤其是 RAG 检索、Agent 循环、MCP 调用）需要写中文注释，说明「为什么这么做」。
- 不做过度抽象：没有第二个调用方之前，不要提前抽取基类或工厂模式。

## 工作约定（重要）

1. **不要随意修改无关文件。** 只改动与当前任务直接相关的文件。
2. **修改代码前先分析现有代码结构**，理解清楚现状再动手，不要凭猜测改代码。
3. **每完成一个功能，都要说明：**
   - 修改/新增了哪些文件
   - 每个文件为什么这样改（而不是只列文件名）
4. **不要创建不必要的文件。** 新增文件前先确认它是否有明确职责。
5. 引入新的第三方依赖时，需要说明选它的理由，并同步更新 `requirements.txt`。

## 开发路线（建议顺序）

1. 项目骨架 + 配置管理（环境变量、日志）
2. PostgreSQL 接入 + 数据模型与迁移
3. DeepSeek 接入，打通最基础的对话接口
4. 文档上传与切分、向量化入库（Milvus）
5. RAG 检索问答链路
6. Agent 工具调用与多步推理
7. MCP 客户端接入外部工具
8. Vue 3 前端界面
9. Docker Compose 一键部署 + README 文档
