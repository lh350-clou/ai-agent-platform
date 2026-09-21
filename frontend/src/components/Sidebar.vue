<script setup lang="ts">
import KnowledgeBaseList from './KnowledgeBaseList.vue'
import type { KnowledgeBase } from '../types'

defineProps<{
  knowledgeBases: KnowledgeBase[]
  selectedId: string | null
  loading: boolean
  error: string | null
}>()

const emit = defineEmits<{
  (e: 'select', id: string): void
  (e: 'create'): void
  (e: 'delete', knowledgeBase: KnowledgeBase): void
}>()
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar__brand">
      <span class="sidebar__logo" aria-hidden="true">AI</span>
      <h1 class="sidebar__title">AI 智能知识库 Agent 平台</h1>
    </div>

    <button class="sidebar__create" type="button" @click="emit('create')">
      + 新建知识库
    </button>

    <nav class="sidebar__nav" aria-label="知识库列表">
      <p class="sidebar__section-title">知识库</p>
      <KnowledgeBaseList
        :items="knowledgeBases"
        :selected-id="selectedId"
        :loading="loading"
        :error="error"
        @select="emit('select', $event)"
        @delete="emit('delete', $event)"
      />
    </nav>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
  /* 固定宽度 + flex-shrink: 0：侧栏不参与「剩余空间」的分配，
     窗口变窄时由右侧内容先收缩，而不是把侧栏压变形。 */
  width: var(--sidebar-width);
  flex-shrink: 0;
  padding: 20px 14px;
  background-color: #fff;
  border-right: 1px solid var(--color-border);
  overflow-y: auto;
}

.sidebar__brand {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 0 6px;
}

.sidebar__logo {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  flex-shrink: 0;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  background-color: var(--color-accent);
  border-radius: 8px;
}

.sidebar__title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.35;
}

.sidebar__create {
  padding: 9px 12px;
  font: inherit;
  font-size: 13px;
  font-weight: 500;
  color: #fff;
  background-color: var(--color-accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.sidebar__create:hover {
  background-color: var(--color-accent-strong);
}

.sidebar__nav {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sidebar__section-title {
  margin: 0;
  padding: 0 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  color: var(--color-text-faint);
  text-transform: uppercase;
}
</style>
