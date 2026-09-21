<script setup lang="ts">
import { ref, watch } from 'vue'

// 这个组件只管收集输入，不自己发请求 ——
// 「创建成功之后要刷新列表、要选中新库」是页面的职责，
// 模态框知道得越少，越能复用到别的地方。
const props = defineProps<{
  open: boolean
  submitting: boolean
  error: string | null
}>()

const emit = defineEmits<{
  (e: 'submit', name: string, description: string): void
  (e: 'close'): void
}>()

const name = ref('')
const description = ref('')

// 每次打开都清空上一次的输入。
// 不清的话，上次失败后残留的内容会让人以为「已经填过了」，
// 而第二次打开时又分不清那是新输入还是旧残留。
watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      name.value = ''
      description.value = ''
    }
  },
)

function submit(): void {
  if (props.submitting) return
  if (name.value.trim().length === 0) return
  emit('submit', name.value.trim(), description.value.trim())
}
</script>

<template>
  <div v-if="open" class="overlay" @click.self="emit('close')">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="create-kb-title">
      <h2 id="create-kb-title" class="modal__title">新建知识库</h2>

      <form class="modal__form" @submit.prevent="submit">
        <label class="field">
          <span class="field__label">名称</span>
          <input
            v-model="name"
            class="field__input"
            type="text"
            maxlength="255"
            placeholder="例如：产品文档"
            :disabled="submitting"
            autofocus
          />
        </label>

        <label class="field">
          <span class="field__label">描述<span class="field__optional">（可选）</span></span>
          <textarea
            v-model="description"
            class="field__input field__input--area"
            maxlength="2000"
            rows="3"
            placeholder="这个知识库收录什么内容"
            :disabled="submitting"
          ></textarea>
        </label>

        <!-- 失败时错误显示在表单里、模态框不关：
             关掉的话用户填的内容就没了，得重填一遍。 -->
        <p v-if="error" class="modal__error">{{ error }}</p>

        <div class="modal__actions">
          <button
            class="btn btn--ghost"
            type="button"
            :disabled="submitting"
            @click="emit('close')"
          >
            取消
          </button>
          <button class="btn btn--primary" type="submit" :disabled="submitting || !name.trim()">
            {{ submitting ? '创建中…' : '创建' }}
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 24px;
  background-color: rgba(15, 23, 42, 0.4);
  z-index: 50;
}

.modal {
  width: 100%;
  max-width: 420px;
  padding: 22px;
  background-color: #fff;
  border-radius: 14px;
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.18);
}

.modal__title {
  margin: 0 0 16px;
  font-size: 17px;
  font-weight: 600;
}

.modal__form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field__label {
  font-size: 13px;
  font-weight: 500;
}

.field__optional {
  font-weight: 400;
  color: var(--color-text-faint);
}

.field__input {
  padding: 9px 12px;
  font: inherit;
  font-size: 14px;
  color: inherit;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.field__input:focus {
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px var(--color-accent-soft);
}

.field__input--area {
  resize: vertical;
}

.modal__error {
  margin: 0;
  padding: 8px 11px;
  font-size: 13px;
  color: #b42318;
  background-color: #fef3f2;
  border-radius: 8px;
}

.modal__actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 2px;
}

.btn {
  padding: 9px 18px;
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.btn--primary {
  color: #fff;
  background-color: var(--color-accent);
}

.btn--primary:hover:not(:disabled) {
  background-color: var(--color-accent-strong);
}

.btn--ghost {
  color: var(--color-text-muted);
  background-color: var(--color-surface-hover);
}

.btn--ghost:hover:not(:disabled) {
  background-color: #e5e7eb;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
