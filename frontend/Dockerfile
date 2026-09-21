# ============================================================
# 前端镜像：Node 构建 → Nginx 提供静态文件
#
# 分两阶段是因为构建和运行需要的完全不是一套东西：
# 构建要 Node + node_modules（几百 MB），运行时只要几个静态文件。
# 合成一个阶段的话，最终镜像里会带着整套 Node 环境和 node_modules。
# ============================================================

# ---------- 阶段一：构建 ----------
FROM node:22-alpine AS builder

WORKDIR /app

# 先装依赖再拷源码，让依赖层能命中缓存（同后端 Dockerfile 的思路）。
# 用 npm ci 而不是 npm install：ci 严格按 package-lock.json 安装，
# 保证构建出来的产物和本地验证过的版本完全一致；
# install 则可能因为解析到新版本而装出不一样的东西。
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# 后端地址在【构建时】注入。
#
# Vite 会把 import.meta.env.VITE_* 在打包时替换成字面量，所以这个值
# 是烧进产物的，运行时再改环境变量没用 —— 要换地址得重新构建。
#
# 默认值是 http://localhost:8000：浏览器访问的是宿主机，不是容器，
# 所以这里必须写 localhost，而不是 compose 里的服务名 backend。
# 写了 backend 的话，浏览器会去解析这个主机名，然后失败。
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ---------- 阶段二：运行 ----------
FROM nginx:alpine

# 只把构建产物拷过来，不带 Node、不带源码、不带 node_modules
COPY --from=builder /app/dist /usr/share/nginx/html

# 覆盖默认站点配置（加了 history 路由回退，见 nginx.conf 里的说明）
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
