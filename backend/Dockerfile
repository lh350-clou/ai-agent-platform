# ============================================================
# 后端镜像：FastAPI + 迁移 + MCP Server（stdio 子进程）
#
# 用 python:3.12-slim 而不是 3.14：
#   1. 3.12 是目前各生态（pymilvus / grpcio / asyncpg）轮子最齐的版本，
#      新版本 Python 往往要等依赖跟上；
#   2. 本机开发用的是 3.14，但那是 Anaconda 环境 —— 镜像里跑 3.12
#      不影响本地开发，只是多一个可选的部署方式。
# ============================================================
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE：容器里不需要 .pyc，写了也只是多占一层空间
# PYTHONUNBUFFERED：让日志实时出现在 docker logs 里。
#   不加的话 stdout 会被缓冲，出问题时 docker logs 看不到最后几行 ——
#   而那几行往往正是报错原因。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 注意这个层级：代码必须落在 /app/backend/app/... 而不是 /app/app/...
#
# 因为 config.py 用 Path(__file__).resolve().parents[3] 反推「仓库根目录」，
# 它同时决定了三件事：
#   - .env 从哪读
#   - 上传文件落在哪（<根>/storage/documents）
#   - MCP Server 的工作目录（<根>/backend）
# 把代码放在 /app/backend/ 下，parents[3] 正好是 /app，
# 于是 storage 落在 /app/storage —— 正好是下面挂卷的路径。
# 如果改成 /app/app/...，storage 会跑到 /storage，挂载就对不上了。
WORKDIR /app/backend

# 先只拷依赖清单再装依赖：只要 requirements.txt 没变，
# 这一层就能命中缓存，改业务代码时不用重新装一遍依赖（几分钟的事）。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷其余代码
COPY . .

EXPOSE 8000

# 注意这里没有 CMD —— 启动命令写在 docker-compose.yml 里，
# 因为要带上前置的 `alembic upgrade head`。
# 放在 compose 里也方便不重建镜像就调整启动参数。
