<script setup lang="ts">
import type { KnowledgeBase } from '../types'

defineProps<{
  items: KnowledgeBase[]
  selectedId: string | null
  loading: boolean
  error: string | null
}>()

const emit = defineEmits<{
  (e: 'select', id: string): void
  (e: 'delete', knowledgeBase: KnowledgeBase): void
}>()
</script>

<template>
  <!-- 加载中：只在还没有数据时显示整块提示。
       刷新时列表已经有内容了，用整块 loading 替换会让界面闪一下。 -->
  <p v-if="loading && items.length === 0" class="kb-state">正在加载…</p>

  <!-- 加载失败：明确告诉用户失败了，而不是显示「还没有知识库」——
       后者会让人以为数据库真的是空的，从而去做一些没必要的操作。 -->
  <p v-else-if="error" class="kb-state kb-state--error">{{ error }}</p>

  <ul v-else-if="items.length > 0" class="kb-list">
    <li v-for="kb in items" :key="kb.id" class="kb-row">
      <button
        class="kb-item"
        :class="{ 'kb-item--active': kb.id === selectedId }"
        type="button"
        :aria-current="kb.id === selectedId ? 'true' : undefined"
        @click="emit('select', kb.id)"
      >
        <span class="kb-item__name">{{ kb.name }}</span>
        <span class="kb-item__desc">{{ kb.description || '暂无描述' }}</span>
        <span class="kb-item__count">{{ kb.document_count }} 篇文档</span>
      </button>

      <!-- 删除按钮只在选中时出现：一行里常驻两个按钮会让列表显得很吵，
           而删除又是破坏性操作，不该在未选中时随手可点。 -->
      <button
        v-if="kb.id === selectedId"
        class="kb-row__delete"
        type="button"
        title="删除这个知识库"
        aria-label="删除知识库"
        @click.stop="emit('delete', kb)"
      >
        ×
      </button>
    </li>
  </ul>

  <p v-else class="kb-state">还没有知识库</p>
</template>

<style scoped>
.kb-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.kb-row {
  position: relative;
}

.kb-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 100%;
  padding: 9px 11px;
  font: inherit;
  text-align: left;
  color: var(--color-text);
  background-color: transparent;
  border: 1px solid transparent;
  /* 左侧留一条透明边框：选中时它会变成强调色。
     始终占位是为了让选中前后文字不左右跳动。 */
  border-left: 3px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.15s, border-color 0.15s;
}

.kb-item:hover {
  background-color: var(--color-surface-hover);
}

/* 选中态用「背景 + 左侧色条 + 文字加粗」三重强调，
   只改背景色的话在浅色主题下对比不够明显。 */
.kb-item--active {
  background-color: var(--color-surface-active);
  border-left-color: var(--color-accent);
  /* 给右侧的删除按钮让出位置，避免长名称被按钮压住 */
  padding-right: 30px;
}

.kb-item--active .kb-item__name {
  font-weight: 600;
  color: var(--color-accent-strong);
}

.kb-item__name {
  font-size: 14px;
  /* 名称过长时省略而不是换行 —— 列表里每个条目高度一致才整齐 */
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-item__desc {
  font-size: 12px;
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-item__count {
  margin-top: 2px;
  font-size: 11px;
  color: var(--color-text-faint);
}

.kb-row__delete {
  position: absolute;
  top: 8px;
  right: 8px;
  display: grid;
  place-items: center;
  width: 20px;
  height: 20px;
  padding: 0;
  font-size: 15px;
  line-height: 1;
  color: var(--color-text-faint);
  background-color: transparent;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.kb-row__delete:hover {
  color: #b42318;
  background-color: #fef3f2;
}

.kb-state {
  margin: 0;
  padding: 12px;
  font-size: 13px;
  color: var(--color-text-muted);
  text-align: center;
}

.kb-state--error {
  color: #b42318;
  text-align: left;
  overflow-wrap: anywhere;
}
</style>
