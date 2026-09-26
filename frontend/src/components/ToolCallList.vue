<script setup lang="ts">
import type { AgentToolCall } from '../types'

/**
 * 本次回答执行过的工具调用。
 *
 * 这里【不显示】每次调用的成功/失败和耗时：后端确实记录了它们，但只记在
 * 服务端的 Trace 里，没有暴露到 AgentToolCall 这个结构上（见
 * backend/app/schemas/agent.py）。所以界面上只呈现后端真正给出来的东西 ——
 * 工具名和参数。想知道调用的成败，看同一条回答下的「运行记录」里模型调用的状态，
 * 或者拿 trace_id 去服务端日志里查。
 */
defineProps<{
  toolCalls: AgentToolCall[]
}>()

/**
 * 把参数渲染成一行可读文本，例如 "query: Milvus 是什么 · top_k: 3"。
 *
 * 只渲染后端返回的字段，不做任何推断或补全：
 * 这些内容会直接展示给用户，而前端不该「猜」后端执行了什么。
 */
function formatArguments(args: Record<string, unknown>): string {
  const entries = Object.entries(args)
  if (entries.length === 0) return '无参数'

  return entries
    .map(([key, value]) => {
      // 非字符串（数字、布尔）直接转字符串；对象/数组用 JSON 表示。
      // 这里的值都来自后端返回的工具参数，结构很浅，不会出现深层嵌套。
      const text = typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)
      return `${key}: ${text}`
    })
    .join(' · ')
}
</script>

<template>
  <div v-if="toolCalls.length > 0" class="tools">
    <p class="tools__title">已调用工具</p>
    <ul class="tools__list">
      <li v-for="(call, index) in toolCalls" :key="`${call.tool}-${index}`" class="tools__item">
        <code class="tools__name">{{ call.tool }}</code>
        <span class="tools__args">{{ formatArguments(call.arguments) }}</span>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.tools {
  margin-top: 8px;
  padding: 9px 12px;
  background-color: #f8fafc;
  border: 1px dashed var(--color-border);
  border-radius: 8px;
}

.tools__title {
  margin: 0 0 5px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--color-text-faint);
  text-transform: uppercase;
}

.tools__list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.tools__item {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: baseline;
  font-size: 12px;
}

.tools__name {
  padding: 1px 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--color-accent-strong);
  background-color: var(--color-surface-active);
  border-radius: 4px;
}

.tools__args {
  color: var(--color-text-muted);
  /* 参数可能很长，允许在任意位置断行，避免把气泡撑宽 */
  overflow-wrap: anywhere;
}
</style>
