<script setup lang="ts">
import { computed, ref } from 'vue'

import { formatDateTime } from '../format'
import type { DocumentStatus, KnowledgeBase, KnowledgeDocument } from '../types'

/**
 * 文档面板：上传 TXT、查看状态、删除。
 *
 * 一个必须说清楚的前提：后端【没有】文档列表接口。
 * 所以这里的列表不是「这个知识库的全部文档」，而是「本机上传过的文档」——
 * ID 存在 localStorage 里，每次打开都按 ID 向后端重新查一遍真实状态。
 * 界面上必须把这件事写明白（见模板里的提示文字），
 * 否则用户会理所当然地把它当成完整列表，然后困惑于「总数 3 篇，列表怎么只有 1 条」。
 */
const props = defineProps<{
  /** 当前选中的知识库。为 null 表示还没选。 */
  knowledgeBase: KnowledgeBase | null
  documents: KnowledgeDocument[]
  /** 正在按 ID 逐个向后端查最新状态。 */
  loading: boolean
  /** 查询失败的原因（上传失败走 uploadError，两者分开以免互相覆盖）。 */
  error: string | null
  uploading: boolean
  uploadError: string | null
  /** 正在删除的文档 ID，用来只禁用那一行的按钮。 */
  deletingId: string | null
}>()

const emit = defineEmits<{
  (e: 'upload', file: File): void
  (e: 'delete', document: KnowledgeDocument): void
  (e: 'refresh'): void
}>()

/** 状态的中文说明。后端返回的是英文枚举值，直接显示出来对用户没有意义。 */
const STATUS_TEXT: Record<DocumentStatus, string> = {
  pending: '排队中',
  processing: '处理中',
  completed: '已完成',
  failed: '失败',
}

const fileInput = ref<HTMLInputElement | null>(null)
/** 本地校验的提示（比如选了非 .txt 文件），与后端返回的错误分开展示。 */
const localError = ref<string | null>(null)

// 本地提示优先：它针对的是「刚刚这次操作」，比上一次后端错误更贴近当前动作。
const displayError = computed(() => localError.value ?? props.uploadError)

function statusText(status: DocumentStatus): string {
  return STATUS_TEXT[status] ?? status
}

function pickFile(): void {
  localError.value = null
  fileInput.value?.click()
}

function handleFileChange(event: Event): void {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  // 选完立刻清空 input 的值：不清的话，用户第二次选同一个文件不会触发 change 事件，
  // 表现成「点了没反应」。（同一个文件重传是常见操作 —— 比如上次入库失败了。）
  input.value = ''
  if (!file) return

  // 这里查扩展名只是为了早点给出提示，不是安全边界 ——
  // 真正的校验在后端（它会重新判断扩展名、编码和大小）。
  // 加这一道是因为后端的报错要等一次完整的上传往返，本地的能立刻显示。
  if (!file.name.toLowerCase().endsWith('.txt')) {
    localError.value = `目前只支持 .txt 文件，选中的是：${file.name}`
    return
  }

  emit('upload', file)
}
</script>

<template>
  <section class="docs">
    <header class="docs__header">
      <div class="docs__heading">
        <h2 class="docs__title">文档</h2>
        <!-- 文档总数用知识库接口实时统计出来的值，它是权威的；
             下面列表的条数只是「本机记录」，两者可能不一致，所以分开写。 -->
        <p v-if="knowledgeBase" class="docs__subtitle">
          「{{ knowledgeBase.name }}」· 后端统计共 {{ knowledgeBase.document_count }} 篇文档
        </p>
      </div>
      <button
        class="docs__refresh"
        type="button"
        :disabled="!knowledgeBase || loading"
        @click="emit('refresh')"
      >
        {{ loading ? '刷新中…' : '刷新' }}
      </button>
    </header>

    <div class="docs__body">
      <div v-if="!knowledgeBase" class="docs__placeholder">
        <p class="docs__placeholder-main">选择一个知识库，上传并管理它的文档</p>
        <p class="docs__placeholder-sub">从左侧列表中选择，或新建一个知识库</p>
      </div>

      <template v-else>
        <div class="uploader">
          <input
            ref="fileInput"
            class="uploader__input"
            type="file"
            accept=".txt,text/plain"
            @change="handleFileChange"
          />
          <button
            class="uploader__button"
            type="button"
            :disabled="uploading"
            @click="pickFile"
          >
            {{ uploading ? '上传并入库中…' : '选择 TXT 文件上传' }}
          </button>
          <!-- 说明「会等多久」：上传接口是同步入库的，大文件要等几十秒，
               提前讲清楚，用户才不会以为卡死了。 -->
          <p class="uploader__hint">
            只支持 UTF-8 编码的 .txt 文件。上传后后端会同步完成切分与向量化，
            文件较大时需要等待几十秒，请勿重复点击。
          </p>
        </div>

        <p v-if="displayError" class="docs__error">{{ displayError }}</p>

        <p class="docs__note">
          列表只包含本机上传过的文档（后端暂未提供文档列表接口）。
          文档的状态与切片数每次都向后端实时查询，不是本地缓存的旧值。
        </p>

        <!-- 三种状态分开表达，理由同知识库列表：
             把「加载失败」显示成「还没有上传」会让人白跑一趟。 -->
        <p v-if="loading && documents.length === 0" class="docs__state">正在加载…</p>
        <p v-else-if="error" class="docs__state docs__state--error">{{ error }}</p>
        <p v-else-if="documents.length === 0" class="docs__state">
          本机还没有上传过文档
        </p>

        <ul v-else class="doc-list">
          <li v-for="doc in documents" :key="doc.id" class="doc">
            <div class="doc__main">
              <span class="doc__name">{{ doc.name }}</span>
              <span class="doc__meta">
                <span class="doc__status" :class="`doc__status--${doc.status}`">
                  {{ statusText(doc.status) }}
                </span>
                <span v-if="doc.status === 'completed'">{{ doc.chunk_count }} 个切片</span>
                <span>{{ formatDateTime(doc.created_at) }}</span>
                <!-- 显示 ID 前 8 位：RAG 的检索来源里给的是完整 document_id，
                     有这一段就能把「引用来自哪篇文档」对起来。 -->
                <code class="doc__id" :title="doc.id">{{ doc.id.slice(0, 8) }}</code>
              </span>
              <p v-if="doc.error_message" class="doc__error">{{ doc.error_message }}</p>
            </div>

            <button
              class="doc__delete"
              type="button"
              :disabled="deletingId === doc.id"
              @click="emit('delete', doc)"
            >
              {{ deletingId === doc.id ? '删除中…' : '删除' }}
            </button>
          </li>
        </ul>
      </template>
    </div>
  </section>
</template>

<style scoped>
.docs {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-width: 0;
  background-color: var(--color-bg);
}

.docs__header {
  display: flex;
  flex-shrink: 0;
  gap: 16px;
  align-items: flex-start;
  justify-content: space-between;
  padding: 16px 24px;
  background-color: #fff;
  border-bottom: 1px solid var(--color-border);
}

.docs__heading {
  min-width: 0;
}

.docs__title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.docs__subtitle {
  margin: 3px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.docs__refresh {
  flex-shrink: 0;
  padding: 7px 14px;
  font: inherit;
  font-size: 13px;
  color: var(--color-text-muted);
  background-color: var(--color-surface-hover);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.docs__refresh:hover:not(:disabled) {
  background-color: #e5e7eb;
}

.docs__refresh:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.docs__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 20px 24px 24px;
}

.docs__placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 100%;
  text-align: center;
}

.docs__placeholder-main {
  margin: 0;
  font-size: 15px;
  color: var(--color-text-muted);
}

.docs__placeholder-sub {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-faint);
}

.uploader {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-start;
  padding: 16px;
  background-color: #fff;
  border: 1px dashed var(--color-border);
  border-radius: 12px;
}

/* 原生 file input 的样式在各浏览器里差别很大且很难改，
   这里把它藏起来，用下面的按钮代为触发（点击仍然走它的原生行为）。 */
.uploader__input {
  display: none;
}

.uploader__button {
  padding: 9px 18px;
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  background-color: var(--color-accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.uploader__button:hover:not(:disabled) {
  background-color: var(--color-accent-strong);
}

.uploader__button:disabled {
  background-color: #c7cbd4;
  cursor: not-allowed;
}

.uploader__hint {
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-faint);
}

.docs__error {
  margin: 12px 0 0;
  padding: 9px 12px;
  font-size: 13px;
  color: #b42318;
  background-color: #fef3f2;
  border-radius: 8px;
  overflow-wrap: anywhere;
}

.docs__note {
  margin: 14px 0 0;
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-faint);
}

.docs__state {
  margin: 16px 0 0;
  padding: 16px;
  font-size: 13px;
  color: var(--color-text-muted);
  text-align: center;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 12px;
}

.docs__state--error {
  color: #b42318;
  background-color: #fef3f2;
  border-color: #fecdca;
  text-align: left;
  overflow-wrap: anywhere;
}

.doc-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 14px 0 0;
  padding: 0;
  list-style: none;
}

.doc {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
  padding: 12px 14px;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 10px;
}

.doc__main {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.doc__name {
  font-size: 14px;
  font-weight: 500;
  overflow-wrap: anywhere;
}

.doc__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: baseline;
  font-size: 12px;
  color: var(--color-text-muted);
}

.doc__status {
  padding: 1px 7px;
  font-size: 11px;
  border-radius: 4px;
}

/* 状态用底色区分，同时模板里有文字说明 —— 不单靠颜色传达信息。 */
.doc__status--completed {
  color: #067647;
  background-color: #ecfdf3;
}

.doc__status--processing,
.doc__status--pending {
  color: #b54708;
  background-color: #fffaeb;
}

.doc__status--failed {
  color: #b42318;
  background-color: #fef3f2;
}

.doc__id {
  padding: 1px 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--color-text-faint);
  background-color: var(--color-surface-hover);
  border-radius: 4px;
}

.doc__error {
  margin: 0;
  font-size: 12px;
  color: #b42318;
  overflow-wrap: anywhere;
}

.doc__delete {
  flex-shrink: 0;
  padding: 6px 12px;
  font: inherit;
  font-size: 12px;
  color: #b42318;
  background-color: transparent;
  border: 1px solid #fecdca;
  border-radius: 7px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.doc__delete:hover:not(:disabled) {
  background-color: #fef3f2;
}

.doc__delete:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
