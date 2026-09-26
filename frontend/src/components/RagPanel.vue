<script setup lang="ts">
import { ref } from 'vue'

import ChatInput from './ChatInput.vue'
import { formatScore } from '../format'
import type { AskResponse, KnowledgeBase } from '../types'

/**
 * RAG 问答面板：固定「先检索、再作答」。
 *
 * 和 Agent 面板的区别是模型没有选择权：这条路每次都先检索，再把命中的资料
 * 连同问题交给模型。所以这里展示的重点除了回答，还有【用到的是哪几段资料】——
 * 那是 RAG 和普通对话最本质的区别，也是排查「回答为什么不对」时的第一手线索。
 */
const props = defineProps<{
  knowledgeBase: KnowledgeBase | null
  asking: boolean
  error: string | null
  /** 上一次问答的结果。切换知识库时为 null。 */
  result: AskResponse | null
}>()

const emit = defineEmits<{
  (e: 'ask', question: string, topK: number): void
}>()

/**
 * 检索条数。
 *
 * 三个值与后端 app/schemas/qa.py 保持一致：默认 5，范围 1~10。
 * 上限比检索接口（20）紧，是因为这里每条结果都会被拼进提示词送给模型 ——
 * 条数越多，上下文越长、越慢，无关内容也越容易干扰模型。
 */
const TOP_K_MIN: number = 1
const TOP_K_MAX: number = 10
const TOP_K_DEFAULT: number = 5
const topK = ref<number>(TOP_K_DEFAULT)

function submit(text: string): void {
  if (!props.knowledgeBase || props.asking) return
  emit('ask', text, topK.value)
}
</script>

<template>
  <section class="rag">
    <header class="rag__header">
      <h2 class="rag__title">RAG 问答</h2>
      <p class="rag__subtitle">
        <template v-if="knowledgeBase">
          「{{ knowledgeBase.name }}」· 每次提问都先在这个知识库里检索，再由模型基于检索结果作答
        </template>
        <template v-else>选择一个知识库，基于它的内容提问</template>
      </p>
    </header>

    <div class="rag__body">
      <div v-if="!knowledgeBase" class="rag__placeholder">
        <p class="rag__placeholder-main">选择一个知识库，基于它的内容提问</p>
        <p class="rag__placeholder-sub">从左侧列表中选择，或新建一个知识库</p>
      </div>

      <p v-else-if="asking" class="rag__placeholder-main">正在检索并生成回答…</p>

      <div v-else-if="result" class="rag__result">
        <p class="rag__question">{{ result.question }}</p>
        <p class="rag__answer">{{ result.answer }}</p>

        <div class="sources">
          <p class="sources__title">
            检索来源（{{ result.sources.length }} 条）
          </p>

          <!-- 空和「有」必须分开说：没有检索到资料时模型仍会给一个回答，
               那个回答的可信度完全不同，不能让人误以为有出处。 -->
          <p v-if="result.sources.length === 0" class="sources__empty">
            本次没有检索到相关资料，上面的回答没有可引用的出处。
            可以先在「文档」里上传内容，或换一种问法。
          </p>

          <ol v-else class="source-list">
            <li v-for="(source, index) in result.sources" :key="source.chunk_id" class="source">
              <div class="source__meta">
                <span class="source__index">#{{ index + 1 }}</span>
                <span class="source__score">相似度 {{ formatScore(source.score) }}</span>
                <code class="source__id" :title="source.document_id">
                  文档 {{ source.document_id.slice(0, 8) }}
                </code>
                <code class="source__id" :title="source.chunk_id">
                  {{ source.chunk_id }}
                </code>
              </div>
              <p class="source__content">{{ source.content }}</p>
            </li>
          </ol>
        </div>
      </div>

      <div v-else class="rag__placeholder">
        <p class="rag__placeholder-main">还没有提问</p>
        <p class="rag__placeholder-sub">
          在下方输入问题，答案会连同用到的资料一起显示出来
        </p>
      </div>
    </div>

    <div class="rag__options">
      <label class="rag__option">
        检索条数
        <select v-model.number="topK" class="rag__select">
          <option v-for="n in TOP_K_MAX - TOP_K_MIN + 1" :key="n" :value="n + TOP_K_MIN - 1">
            {{ n + TOP_K_MIN - 1 }}
          </option>
        </select>
      </label>
      <span class="rag__option-hint">
        条数越多，拼给模型的上下文越长、响应越慢（后端上限 {{ TOP_K_MAX }}）
      </span>
    </div>

    <p v-if="error" class="rag__error">{{ error }}</p>

    <!-- 复用聊天输入框：它只负责「收集一句话并抛出去」，
         对 RAG 来说这个职责完全一样，没必要再写一个。 -->
    <ChatInput
      :disabled="!knowledgeBase || asking"
      :placeholder="asking ? '正在检索并生成回答…' : '输入问题，Enter 发送，Shift + Enter 换行'"
      @send="submit"
    />
  </section>
</template>

<style scoped>
.rag {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-width: 0;
  background-color: var(--color-bg);
}

.rag__header {
  flex-shrink: 0;
  padding: 16px 24px;
  background-color: #fff;
  border-bottom: 1px solid var(--color-border);
}

.rag__title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.rag__subtitle {
  margin: 3px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.rag__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px;
}

.rag__placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 100%;
  text-align: center;
}

.rag__placeholder-main {
  margin: 0;
  font-size: 15px;
  color: var(--color-text-muted);
}

.rag__placeholder-sub {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-faint);
}

.rag__result {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 860px;
}

.rag__question {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-muted);
}

.rag__answer {
  margin: 0;
  padding: 14px 16px;
  font-size: 14px;
  line-height: 1.7;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 12px;
  /* 保留换行：模型回答里的分段和列表才有意义 */
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.sources {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sources__title {
  margin: 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--color-text-faint);
  text-transform: uppercase;
}

.sources__empty {
  margin: 0;
  padding: 12px 14px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-muted);
  background-color: #fffaeb;
  border: 1px solid #fedf89;
  border-radius: 10px;
}

.source-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.source {
  padding: 12px 14px;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 10px;
}

.source__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: baseline;
  margin-bottom: 6px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.source__index {
  font-weight: 600;
  color: var(--color-accent-strong);
}

.source__score {
  font-variant-numeric: tabular-nums;
}

.source__id {
  padding: 1px 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--color-text-faint);
  background-color: var(--color-surface-hover);
  border-radius: 4px;
  overflow-wrap: anywhere;
}

.source__content {
  margin: 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--color-text);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.rag__options {
  display: flex;
  flex-shrink: 0;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  padding: 10px 24px 0;
  background-color: #fff;
  border-top: 1px solid var(--color-border);
}

.rag__option {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 12px;
  color: var(--color-text-muted);
}

.rag__select {
  padding: 4px 6px;
  font: inherit;
  font-size: 12px;
  color: inherit;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}

.rag__option-hint {
  font-size: 12px;
  color: var(--color-text-faint);
}

.rag__error {
  flex-shrink: 0;
  margin: 0;
  padding: 10px 24px;
  font-size: 13px;
  color: #b42318;
  background-color: #fef3f2;
  border-top: 1px solid #fecdca;
  overflow-wrap: anywhere;
}
</style>
