<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import ChatPanel from '../components/ChatPanel.vue'
import CreateKnowledgeBaseModal from '../components/CreateKnowledgeBaseModal.vue'
import Sidebar from '../components/Sidebar.vue'
import { askAgent } from '../api/agent'
import { ApiError } from '../api/client'
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBases,
} from '../api/knowledgeBase'
import type { ChatMessage, KnowledgeBase } from '../types'

// 这一层是「状态的唯一持有者」：知识库列表、当前选中项、对话内容都在这。
// 子组件只通过 props 拿数据、通过事件往回抛意图，自己不保存状态 ——
// 状态散在多个组件里时，最容易出现两处显示不一致的问题。
const knowledgeBases = ref<KnowledgeBase[]>([])
const selectedId = ref<string | null>(null)
const messagesByKb = ref<Record<string, ChatMessage[]>>({})

// 每个知识库各自对应一个会话 ID。
//
// 按知识库分开存，而不是只留「当前会话」一个变量：会话是【属于某个知识库】的，
// 用单个变量的话，从 A 库切到 B 库再发消息时，很容易把 A 的 conversation_id
// 带过去 —— 后端会直接拒绝（会话不属于这个库），但更糟的是「带过去了却没被拒绝」
// 的情况：那意味着 A 库的对话内容会进入 B 库的上下文，是实打实的数据串台。
// 按 kbId 索引之后，这种错误在结构上就不可能发生。
const conversationIdsByKb = ref<Record<string, string>>({})

const loadingList = ref(false)
const listError = ref<string | null>(null)

const modalOpen = ref(false)
const creating = ref(false)
const createError = ref<string | null>(null)

const thinking = ref(false)
const chatError = ref<string | null>(null)

const selectedKnowledgeBase = computed<KnowledgeBase | null>(
  () => knowledgeBases.value.find((kb) => kb.id === selectedId.value) ?? null,
)

const currentMessages = computed<ChatMessage[]>(() =>
  selectedId.value ? (messagesByKb.value[selectedId.value] ?? []) : [],
)

/**
 * 把任意异常转成能给用户看的一句话。
 *
 * api 层抛的都是 ApiError，消息本身就是中文可读的；其余异常
 * （比如代码里的 TypeError）不该把原始信息展示出去 —— 那既吓人又无用。
 */
function toUserMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return '操作失败，请稍后重试'
}

async function loadKnowledgeBases(): Promise<void> {
  loadingList.value = true
  listError.value = null
  try {
    knowledgeBases.value = await listKnowledgeBases()

    // 选中的库如果已经不在列表里（比如被删了），清掉选中状态，
    // 否则右侧会一直显示一个已经不存在的库。
    if (selectedId.value && !knowledgeBases.value.some((kb) => kb.id === selectedId.value)) {
      selectedId.value = null
    }
  } catch (error) {
    listError.value = toUserMessage(error)
  } finally {
    loadingList.value = false
  }
}

onMounted(loadKnowledgeBases)

function selectKnowledgeBase(id: string): void {
  selectedId.value = id
  // 切换知识库时清掉上一条错误：它是上一个库的上下文，
  // 留着会让人以为新选的这个库也出了问题。
  chatError.value = null
}

function openCreateModal(): void {
  createError.value = null
  modalOpen.value = true
}

function closeCreateModal(): void {
  modalOpen.value = false
}

async function handleCreate(name: string, description: string): Promise<void> {
  creating.value = true
  createError.value = null
  try {
    const created = await createKnowledgeBase({ name, description })

    // 先关弹窗再刷新列表：刷新要等一次网络往返，
    // 弹窗多留那么一下会让人以为「点了没反应」。
    modalOpen.value = false
    await loadKnowledgeBases()
    // 自动选中新建的库 —— 刚建完下一步多半就是要往里传东西或提问。
    selectedId.value = created.id
  } catch (error) {
    // 失败时【不关弹窗】：用户填的内容还在，改一下就能重试。
    createError.value = toUserMessage(error)
  } finally {
    creating.value = false
  }
}

async function handleDelete(knowledgeBase: KnowledgeBase): Promise<void> {
  // 删除不可逆，且会连带删掉这个库下的文档、会话和向量数据。
  const confirmed = window.confirm(
    `确定删除知识库「${knowledgeBase.name}」吗？\n` +
      `它的 ${knowledgeBase.document_count} 篇文档、对话记录和向量数据都会一并删除，且无法恢复。`,
  )
  if (!confirmed) return

  try {
    await deleteKnowledgeBase(knowledgeBase.id)
    // 清理本地的对话缓存和会话 ID：后端的数据已经没了，留着只会在
    // 重新选中同 ID 时显示一堆实际上已被删除的历史；而会话 ID 更是不能留 ——
    // 它指向的记录已随知识库级联删除，再拿去请求会得到 404。
    const { [knowledgeBase.id]: _droppedMessages, ...restMessages } = messagesByKb.value
    messagesByKb.value = restMessages
    const { [knowledgeBase.id]: _droppedConversation, ...restConversations } =
      conversationIdsByKb.value
    conversationIdsByKb.value = restConversations
    if (selectedId.value === knowledgeBase.id) {
      selectedId.value = null
    }
    await loadKnowledgeBases()
  } catch (error) {
    // 删除失败的原因显示在列表区域，和加载失败共用同一个位置 ——
    // 用户此时正在看列表，错误出现在视线范围内才有用。
    listError.value = toUserMessage(error)
  }
}

function appendMessage(kbId: string, message: ChatMessage): void {
  messagesByKb.value = {
    ...messagesByKb.value,
    [kbId]: [...(messagesByKb.value[kbId] ?? []), message],
  }
}

/**
 * 向 Agent 提问。
 *
 * 流程：先把用户消息放进列表（立刻可见），再去请求；拿回答案后再追加
 * assistant 消息。失败时【保留用户消息】—— 那是他确实说过的话，
 * 抹掉会让人怀疑自己到底发出去没有。
 */
async function handleSend(text: string): Promise<void> {
  const kbId = selectedId.value
  // thinking 时直接返回：输入框虽然已经禁用了，但键盘操作或将来
  // 加入别的触发入口时，这道判断是最后一道防线。
  if (!kbId || thinking.value) return

  appendMessage(kbId, {
    id: crypto.randomUUID(),
    role: 'user',
    content: text,
    createdAt: new Date().toISOString(),
  })

  thinking.value = true
  chatError.value = null

  try {
    const response = await askAgent(kbId, {
      question: text,
      // 这个库已经有会话就带上，没有（第一次提问）就不带 ——
      // 后端会新建一个并把 ID 返回。
      conversation_id: conversationIdsByKb.value[kbId],
    })

    // 先记下会话 ID：即使下面追加消息时出错，会话本身也是有效的，
    // 丢掉它会让下一轮提问又开一个新会话，上下文白丢。
    conversationIdsByKb.value = {
      ...conversationIdsByKb.value,
      [kbId]: response.conversation_id,
    }

    appendMessage(kbId, {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: response.answer,
      createdAt: new Date().toISOString(),
      // 把这一轮的工具调用挂在回答上，让用户能看到答案是怎么来的。
      toolCalls: response.tool_calls,
    })
  } catch (error) {
    chatError.value = toUserMessage(error)
  } finally {
    thinking.value = false
  }
}
</script>

<template>
  <div class="layout">
    <Sidebar
      :knowledge-bases="knowledgeBases"
      :selected-id="selectedId"
      :loading="loadingList"
      :error="listError"
      @select="selectKnowledgeBase"
      @create="openCreateModal"
      @delete="handleDelete"
    />
    <main class="layout__main">
      <ChatPanel
        :knowledge-base="selectedKnowledgeBase"
        :messages="currentMessages"
        :thinking="thinking"
        :error="chatError"
        @send="handleSend"
      />
    </main>

    <CreateKnowledgeBaseModal
      :open="modalOpen"
      :submitting="creating"
      :error="createError"
      @submit="handleCreate"
      @close="closeCreateModal"
    />
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  /* 用 100dvh 而不是 100vh：移动端浏览器地址栏会占一部分视口，
     100vh 算的是「地址栏收起时」的高度，会把底部输入框顶出屏幕。 */
  height: 100dvh;
}

.layout__main {
  flex: 1;
  /* min-width: 0 是这条布局里最关键的一行。
     flex 子项默认 min-width: auto，意思是「不能比内容的自然宽度更窄」，
     于是窗口变窄时右侧内容宁可把整体撑出横向滚动条也不肯收缩。
     显式设成 0 之后它才会跟着窗口变窄，左侧栏也就不会被挤变形。 */
  min-width: 0;
}
</style>
