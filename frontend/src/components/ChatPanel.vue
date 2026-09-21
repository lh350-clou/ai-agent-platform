<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'

import ChatInput from './ChatInput.vue'
import ChatMessage from './ChatMessage.vue'
import type { ChatMessage as ChatMessageType, KnowledgeBase } from '../types'

const props = defineProps<{
  /** 当前选中的知识库。为 null 表示还没选。 */
  knowledgeBase: KnowledgeBase | null
  messages: ChatMessageType[]
  /** Agent 正在回答。用来禁用输入并显示等待提示。 */
  thinking: boolean
  /** 上一次提问失败的提示，成功或重发时应被清空。 */
  error: string | null
}>()

const emit = defineEmits<{
  (e: 'send', text: string): void
}>()

const scrollArea = ref<HTMLElement | null>(null)

/**
 * 新消息或「正在思考」出现时滚到底部。
 *
 * 放在 nextTick 里，是因为要等 Vue 把新内容渲染进 DOM 之后才能算出新的高度；
 * 否则滚动位置会停在上一条消息那里，看起来像「消息没出来」。
 */
watch(
  () => [props.messages.length, props.thinking],
  async () => {
    await nextTick()
    scrollArea.value?.scrollTo({ top: scrollArea.value.scrollHeight, behavior: 'smooth' })
  },
)
</script>

<template>
  <section class="chat">
    <header class="chat__header">
      <h2 class="chat__title">{{ knowledgeBase ? knowledgeBase.name : '未选择知识库' }}</h2>
      <p v-if="knowledgeBase" class="chat__subtitle">
        {{ knowledgeBase.description || '暂无描述' }} · {{ knowledgeBase.document_count }} 篇文档
      </p>
    </header>

    <div ref="scrollArea" class="chat__body">
      <!-- 三种状态分开表达：没选库 / 选了但没对话 / 有对话。
           合成一种的话，用户分不清「我还没选」和「这个库是空的」。 -->
      <div v-if="!knowledgeBase" class="chat__placeholder">
        <p class="chat__placeholder-main">选择一个知识库，开始与 AI 对话</p>
        <p class="chat__placeholder-sub">从左侧列表中选择，或新建一个知识库</p>
      </div>

      <div v-else-if="messages.length === 0 && !thinking" class="chat__placeholder">
        <p class="chat__placeholder-main">「{{ knowledgeBase.name }}」还没有对话</p>
        <p class="chat__placeholder-sub">在下方输入问题，开始第一轮对话</p>
      </div>

      <div v-else class="chat__messages">
        <ChatMessage v-for="message in messages" :key="message.id" :message="message" />

        <!-- 「正在思考」也做成一条消息的样子，位置和即将出现的回答一致，
             避免回答出现时整块内容突然下移。 -->
        <div v-if="thinking" class="thinking">
          <div class="thinking__avatar" aria-hidden="true">AI</div>
          <p class="thinking__text">AI 正在思考…</p>
        </div>
      </div>
    </div>

    <!-- 错误条常驻在输入框上方：用户往往是在准备重发时才注意到，
         放在页面顶部或弹 toast 都容易被错过。 -->
    <p v-if="error" class="chat__error">{{ error }}</p>

    <ChatInput
      :disabled="!knowledgeBase || thinking"
      :placeholder="thinking ? 'AI 正在思考…' : '输入问题，Enter 发送，Shift + Enter 换行'"
      @send="emit('send', $event)"
    />
  </section>
</template>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  /* height: 100% 配合父级的 flex 布局，让中间的消息区能独立滚动，
     而头部和输入框固定在上下两端。 */
  height: 100%;
  min-width: 0;
  background-color: var(--color-bg);
}

.chat__header {
  flex-shrink: 0;
  padding: 16px 24px;
  background-color: #fff;
  border-bottom: 1px solid var(--color-border);
}

.chat__title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.chat__subtitle {
  margin: 3px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.chat__body {
  flex: 1;
  /* min-height: 0 让这个可滚动区域真的能滚动：
     默认 min-height: auto 时内容会把容器撑高，滚动条就不出现了。 */
  min-height: 0;
  overflow-y: auto;
  padding: 24px;
}

.chat__messages {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.chat__placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 100%;
  text-align: center;
}

.chat__placeholder-main {
  margin: 0;
  font-size: 15px;
  color: var(--color-text-muted);
}

.chat__placeholder-sub {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-faint);
}

.chat__error {
  flex-shrink: 0;
  margin: 0;
  padding: 10px 24px;
  font-size: 13px;
  color: #b42318;
  background-color: #fef3f2;
  border-top: 1px solid #fecdca;
}

.thinking {
  display: flex;
  gap: 12px;
  align-items: center;
}

.thinking__avatar {
  flex-shrink: 0;
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  background-color: var(--color-accent);
  border-radius: 50%;
}

.thinking__text {
  margin: 0;
  font-size: 14px;
  color: var(--color-text-muted);
}
</style>
