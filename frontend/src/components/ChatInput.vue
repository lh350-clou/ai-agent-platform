<script setup lang="ts">
import { computed, ref } from 'vue'

// 输入框只负责「收集用户输入并抛出去」，自己不碰对话数据 ——
// 它不知道消息会被发到哪、也不知道发完之后会怎样。
// 这样将来换成真正调接口时，这个组件一行都不用改。
const emit = defineEmits<{
  (e: 'send', text: string): void
}>()

const props = withDefaults(
  defineProps<{
    disabled?: boolean
    placeholder?: string
    /**
     * 最多能输入多少字符。
     *
     * 默认 2000，与后端对 question 的上限一致（app/schemas/agent.py 和
     * app/schemas/qa.py 里的 MAX_QUESTION_LENGTH）。超长的问题会被后端以 422
     * 直接拒掉，而那要等一次完整的网络往返才看得到 —— 不如在输入框这一层就挡住。
     */
    maxlength?: number
  }>(),
  {
    disabled: false,
    placeholder: '输入问题，Enter 发送，Shift + Enter 换行',
    maxlength: 2000,
  },
)

const draft = ref('')

// 去掉首尾空白后为空就不让发。用 computed 而不是在提交时判断，
// 是为了让按钮的禁用状态和实际能否发送永远一致 ——
// 两处各判一次的话，迟早会出现「按钮亮着但点了没反应」。
const canSend = computed(() => !props.disabled && draft.value.trim().length > 0)

function submit(): void {
  if (!canSend.value) return
  emit('send', draft.value.trim())
  draft.value = ''
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Enter' || event.shiftKey) return

  // isComposing 是中文输入法的关键：用拼音打字时按 Enter 是「确认候选词」，
  // 不是「发送」。少了这个判断，用户每选一次词都会把半截消息发出去。
  if (event.isComposing) return

  // preventDefault 阻止换行 —— textarea 里 Enter 的默认行为是插入换行，
  // 不挡掉的话「发送」和「换行」会同时发生。
  event.preventDefault()
  submit()
}
</script>

<template>
  <div class="composer">
    <textarea
      v-model="draft"
      class="composer__input"
      :placeholder="placeholder"
      :disabled="disabled"
      :maxlength="maxlength"
      rows="1"
      @keydown="handleKeydown"
    ></textarea>
    <button class="composer__send" type="button" :disabled="!canSend" @click="submit">
      发送
    </button>
  </div>
</template>

<style scoped>
.composer {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 16px 24px 20px;
  border-top: 1px solid var(--color-border);
  background-color: #fff;
}

.composer__input {
  flex: 1;
  /* min-width: 0 是 flex 子项能收缩的前提。
     flex 子项默认 min-width: auto，会被内容撑住不让缩 ——
     结果就是输入框把发送按钮顶出容器。 */
  min-width: 0;
  resize: none;
  padding: 11px 14px;
  font: inherit;
  font-size: 14px;
  line-height: 1.5;
  color: inherit;
  background-color: #fff;
  border: 1px solid var(--color-border);
  border-radius: 10px;
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.composer__input:focus {
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px var(--color-accent-soft);
}

.composer__input:disabled {
  background-color: #f9fafb;
  cursor: not-allowed;
}

.composer__send {
  flex-shrink: 0;
  padding: 11px 22px;
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  background-color: var(--color-accent);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.composer__send:hover:not(:disabled) {
  background-color: var(--color-accent-strong);
}

.composer__send:disabled {
  background-color: #c7cbd4;
  cursor: not-allowed;
}
</style>
