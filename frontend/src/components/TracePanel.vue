<script setup lang="ts">
import { computed } from 'vue'

import { formatDurationMs } from '../format'
import type { AgentTrace } from '../types'

/**
 * 一轮 Agent Run 的运行记录。
 *
 * 展示的字段全部来自 /agent 的响应（trace_id / iterations / llm_calls /
 * total_duration_ms），没有一个是前端推算出来的。刻意不展示的东西：
 *   - 每次【工具调用】的耗时和成败：后端把它们记在服务端的 Trace 里，
 *     但没有暴露到接口上（AgentToolCall 只有工具名和参数）。
 *   - 工具调用次数就直接数响应里的 tool_calls，不另外统计。
 */
const props = defineProps<{
  trace: AgentTrace
  /** 本次回答实际执行过的工具调用次数，来自同一条响应里的 tool_calls。 */
  toolCallCount: number
}>()

const summaryText = computed(
  () =>
    `${props.trace.iterations} 轮 · ${formatDurationMs(props.trace.total_duration_ms)} · ` +
    `${props.trace.llm_calls.length} 次模型调用 · ${props.toolCallCount} 次工具调用`,
)
</script>

<template>
  <!-- 用原生 details/summary 做折叠：默认收起，不喧宾夺主，
       但不需要一行 JS 就能展开 —— 这类「想细看时才看」的信息很适合它。 -->
  <details class="trace">
    <summary class="trace__summary">
      <span class="trace__label">运行记录</span>
      <span class="trace__quick">{{ summaryText }}</span>
    </summary>

    <dl class="trace__facts">
      <div class="trace__fact">
        <dt>Trace ID</dt>
        <!-- 这行要能选中复制：它的用途就是拿去服务端日志里搜同一轮记录。 -->
        <dd class="trace__mono">{{ trace.trace_id }}</dd>
      </div>
      <div class="trace__fact">
        <dt>迭代轮数</dt>
        <dd>{{ trace.iterations }} 轮</dd>
      </div>
      <div class="trace__fact">
        <dt>总耗时</dt>
        <dd>{{ formatDurationMs(trace.total_duration_ms) }}</dd>
      </div>
      <div class="trace__fact">
        <dt>工具调用</dt>
        <dd>{{ toolCallCount }} 次</dd>
      </div>
    </dl>

    <div v-if="trace.llm_calls.length > 0" class="trace__calls">
      <p class="trace__calls-title">模型调用</p>
      <ul class="trace__call-list">
        <li
          v-for="(call, index) in trace.llm_calls"
          :key="`${call.model}-${index}`"
          class="trace__call"
        >
          <span class="trace__call-index">#{{ index + 1 }}</span>
          <code class="trace__mono">{{ call.model }}</code>
          <span class="trace__call-duration">{{ formatDurationMs(call.duration_ms) }}</span>
          <!-- 成敗用文字而不是只靠颜色：色觉差异下红绿并不可靠，
               而且这里失败时还要把原因一并显示出来。 -->
          <span
            class="trace__status"
            :class="call.success ? 'trace__status--ok' : 'trace__status--failed'"
          >
            {{ call.success ? '成功' : '失败' }}
          </span>
          <span v-if="call.error" class="trace__call-error">{{ call.error }}</span>
        </li>
      </ul>
    </div>
  </details>
</template>

<style scoped>
.trace {
  margin-top: 8px;
  padding: 8px 12px;
  background-color: #f8fafc;
  border: 1px dashed var(--color-border);
  border-radius: 8px;
}

.trace__summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: baseline;
  font-size: 12px;
  color: var(--color-text-muted);
  cursor: pointer;
}

.trace__label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--color-text-faint);
  text-transform: uppercase;
}

.trace__quick {
  font-size: 12px;
}

.trace__facts {
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin: 8px 0 0;
}

.trace__fact {
  display: flex;
  gap: 8px;
  font-size: 12px;
}

.trace__fact dt {
  flex-shrink: 0;
  width: 68px;
  color: var(--color-text-faint);
}

.trace__fact dd {
  margin: 0;
  color: var(--color-text-muted);
  overflow-wrap: anywhere;
}

.trace__mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
}

.trace__calls {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--color-border);
}

.trace__calls-title {
  margin: 0 0 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--color-text-faint);
  text-transform: uppercase;
}

.trace__call-list {
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.trace__call {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: baseline;
  font-size: 12px;
  color: var(--color-text-muted);
}

.trace__call-index {
  color: var(--color-text-faint);
}

.trace__call-duration {
  font-variant-numeric: tabular-nums;
}

.trace__status--ok {
  color: #067647;
}

.trace__status--failed {
  color: #b42318;
}

.trace__call-error {
  flex-basis: 100%;
  color: #b42318;
  overflow-wrap: anywhere;
}
</style>
