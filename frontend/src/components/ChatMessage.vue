<script setup lang="ts">
import { computed } from 'vue'

import ToolCallList from './ToolCallList.vue'
import type { ChatMessage } from '../types'

const props = defineProps<{
  message: ChatMessage
}>()

const isUser = computed(() => props.message.role === 'user')

const label = computed(() => (isUser.value ? '我' : 'AI 助手'))

const timeText = computed(() =>
  new Date(props.message.createdAt).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  }),
)
</script>

<template>
  <article class="message" :class="isUser ? 'message--user' : 'message--assistant'">
    <div class="message__avatar" aria-hidden="true">{{ isUser ? '我' : 'AI' }}</div>
    <div class="message__body">
      <header class="message__meta">
        <span class="message__label">{{ label }}</span>
        <time class="message__time">{{ timeText }}</time>
      </header>
      <!-- white-space: pre-wrap（见样式）保留换行，
           否则模型回复里的分段会全部挤成一行。 -->
      <p class="message__content">{{ message.content }}</p>

      <!-- 工具调用只在 assistant 消息下展示：
           用户消息里没有这个概念，显示出来只会让人困惑。
           v-if 里带 length 判断，是为了没有工具调用时不出现一个空框。 -->
      <ToolCallList
        v-if="!isUser && message.toolCalls && message.toolCalls.length > 0"
        :tool-calls="message.toolCalls"
      />
    </div>
  </article>
</template>

<style scoped>
.message {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

/* 用户消息整体靠右，并且把头像放到右侧 —— 这是聊天界面的通用约定，
   不看名字也能一眼分辨谁说的。 */
.message--user {
  flex-direction: row-reverse;
}

.message__avatar {
  flex-shrink: 0;
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  border-radius: 50%;
}

.message--user .message__avatar {
  background-color: #64748b;
}

.message--assistant .message__avatar {
  background-color: var(--color-accent);
}

.message__body {
  /* min-width: 0 让气泡能在长内容下正常收缩换行，而不是把容器撑宽 */
  min-width: 0;
  max-width: 76%;
}

.message--user .message__body {
  /* 用户气泡靠右，文字也跟着右对齐更自然 */
  text-align: right;
}

.message__meta {
  display: flex;
  gap: 8px;
  align-items: baseline;
  margin-bottom: 4px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.message--user .message__meta {
  justify-content: flex-end;
}

.message__label {
  font-weight: 500;
}

.message__content {
  display: inline-block;
  margin: 0;
  padding: 10px 14px;
  font-size: 14px;
  line-height: 1.7;
  text-align: left;
  border-radius: 12px;
  /* 保留换行与空格：模型回复里的分段和缩进才有意义 */
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.message--user .message__content {
  background-color: var(--color-accent);
  color: #fff;
  border-top-right-radius: 4px;
}

.message--assistant .message__content {
  background-color: #fff;
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-top-left-radius: 4px;
}
</style>
