import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue()],

  // 开发服务器配置。
  server: {
    // host 用 0.0.0.0 而不是默认的 localhost：
    // 默认只监听回环地址，容器里跑前端时宿主机就访问不到。
    // 现在虽然还没上 Docker，但把这个默认值先定下来，
    // 免得等做部署时才发现「本地能开、容器里打不开」。
    host: '0.0.0.0',
    port: 5173,
  },

  // 注意：这里【没有】配置 proxy。
  // 前端目前不请求任何后端接口，代理规则等真正开始联调时再加 ——
  // 提前写一份用不上的配置，只会让人以为它已经生效了。
})
