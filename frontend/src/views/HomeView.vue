<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import ChatPanel from '../components/ChatPanel.vue'
import CreateKnowledgeBaseModal from '../components/CreateKnowledgeBaseModal.vue'
import DocumentPanel from '../components/DocumentPanel.vue'
import RagPanel from '../components/RagPanel.vue'
import Sidebar from '../components/Sidebar.vue'
import { askAgent } from '../api/agent'
import { ApiError } from '../api/client'
import { deleteDocument, getDocument, uploadDocument } from '../api/document'
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBases,
} from '../api/knowledgeBase'
import { askKnowledgeBase } from '../api/qa'
import { loadLocalState, saveLocalState } from '../storage'
import type {
  AskResponse,
  ChatMessage,
  ChatSession,
  KnowledgeBase,
  KnowledgeDocument,
} from '../types'

// 这一层是「状态的唯一持有者」：知识库列表、当前选中项、文档、会话、对话内容都在这。
// 子组件只通过 props 拿数据、通过事件往回抛意图，自己不保存状态 ——
// 状态散在多个组件里时，最容易出现两处显示不一致的问题。

/** 主区域的三个功能区。 */
type TabKey = 'documents' | 'rag' | 'agent'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'agent', label: 'Agent 对话' },
  { key: 'rag', label: 'RAG 问答' },
  { key: 'documents', label: '文档' },
]

/** 会话标题的长度上限，与后端 agent.py 里的 TITLE_MAX_LENGTH 一致。 */
const TITLE_MAX_LENGTH: number = 50

const activeTab = ref<TabKey>('agent')

const knowledgeBases = ref<KnowledgeBase[]>([])
const selectedId = ref<string | null>(null)

// ---- 本地状态（localStorage）----
// 同步读一次，不放在 onMounted 里：首屏渲染时数据就已经在手上，
// 否则会先画出「还没有会话」再跳成有内容，闪一下。
// 里面有会话列表、每个库选中的会话、以及本机上传过的文档 ID，
// 之所以要存，是因为后端没有「会话列表」和「文档列表」接口（见 storage.ts）。
const local = loadLocalState()
const sessionsByKb = ref<Record<string, ChatSession[]>>(local.sessions)
const activeSessionByKb = ref<Record<string, string>>(local.activeSession)
const documentIdsByKb = ref<Record<string, string[]>>(local.documentIds)

// ---- 文档 ----
// 只针对【当前选中的知识库】保存一份，不按库分别缓存：
// 面板一次只显示一个库的内容，切换时重新查一遍即可（文档数量通常是个位数）。
const documents = ref<KnowledgeDocument[]>([])
const docsLoading = ref(false)
const docsError = ref<string | null>(null)
const uploading = ref(false)
const uploadError = ref<string | null>(null)
const deletingDocId = ref<string | null>(null)

// ---- RAG ----
const ragResult = ref<AskResponse | null>(null)
const ragAsking = ref(false)
const ragError = ref<string | null>(null)

// ---- 知识库列表 ----
const loadingList = ref(false)
const listError = ref<string | null>(null)

const modalOpen = ref(false)
const creating = ref(false)
const createError = ref<string | null>(null)

// ---- Agent 对话 ----
const thinking = ref(false)
// 三个功能区各有各的错误状态，刻意不共用一个：
// 上传失败的信息不该出现在对话面板里，那会让人以为对话也坏了。
const chatError = ref<string | null>(null)

const selectedKnowledgeBase = computed<KnowledgeBase | null>(
  () => knowledgeBases.value.find((kb) => kb.id === selectedId.value) ?? null,
)

const currentSessions = computed<ChatSession[]>(() =>
  selectedId.value ? (sessionsByKb.value[selectedId.value] ?? []) : [],
)

const activeSessionId = computed<string | null>(() =>
  selectedId.value ? (activeSessionByKb.value[selectedId.value] ?? null) : null,
)

const activeSession = computed<ChatSession | null>(() => {
  const localId = activeSessionId.value
  if (!localId) return null
  return currentSessions.value.find((session) => session.localId === localId) ?? null
})

const currentMessages = computed<ChatMessage[]>(() => activeSession.value?.messages ?? [])

// 本地状态一变就落盘。用 deep watch 而不是在每个修改点手动 save，
// 是因为修改点有好几处（发消息、新建会话、切会话、上传/删除文档），
// 漏掉任何一处的表现都是「刷新后少了点东西」这种很难复现的问题。
watch(
  [sessionsByKb, activeSessionByKb, documentIdsByKb],
  () => {
    saveLocalState({
      sessions: sessionsByKb.value,
      activeSession: activeSessionByKb.value,
      documentIds: documentIdsByKb.value,
    })
  },
  { deep: true },
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

// ---- 会话 ----

function createEmptySession(): ChatSession {
  return {
    localId: crypto.randomUUID(),
    // 还没有后端会话 ID：后端要等第一次提问才会创建会话记录并返回 ID。
    conversationId: null,
    title: '新会话',
    createdAt: new Date().toISOString(),
    messages: [],
  }
}

/** 不可变地改一个会话：生成新的对象和数组，让上面的 deep watch 能察觉到变化。 */
function updateSession(
  kbId: string,
  localId: string,
  updater: (session: ChatSession) => ChatSession,
): void {
  const list = sessionsByKb.value[kbId] ?? []
  sessionsByKb.value = {
    ...sessionsByKb.value,
    [kbId]: list.map((session) => (session.localId === localId ? updater(session) : session)),
  }
}

/** 保证这个知识库至少有一个会话，并且在选中的那个不见了时把选中项接回来。 */
function ensureSession(kbId: string): void {
  const list = sessionsByKb.value[kbId] ?? []

  if (list.length === 0) {
    const session = createEmptySession()
    sessionsByKb.value = { ...sessionsByKb.value, [kbId]: [session] }
    activeSessionByKb.value = { ...activeSessionByKb.value, [kbId]: session.localId }
    return
  }

  if (!list.some((session) => session.localId === activeSessionByKb.value[kbId])) {
    activeSessionByKb.value = { ...activeSessionByKb.value, [kbId]: list[0].localId }
  }
}

function handleNewSession(): void {
  const kbId = selectedId.value
  if (!kbId) return

  const session = createEmptySession()
  // 新会话放在最前面：它一定是用户接下来要用的那个。
  sessionsByKb.value = { ...sessionsByKb.value, [kbId]: [session, ...currentSessions.value] }
  activeSessionByKb.value = { ...activeSessionByKb.value, [kbId]: session.localId }
  chatError.value = null
}

function handleSelectSession(localId: string): void {
  const kbId = selectedId.value
  if (!kbId) return

  activeSessionByKb.value = { ...activeSessionByKb.value, [kbId]: localId }
  // 上一条错误属于上一个会话，留着会让人以为新会话也出了问题。
  chatError.value = null
}

// ---- 知识库 ----

function selectKnowledgeBase(id: string): void {
  selectedId.value = id
}

// 切换知识库时清掉所有「跟着库走」的界面状态。
// 集中在 watch 里做，而不是散在每个切换入口（列表点击、新建后自动选中、删除后清空），
// 少写几处重复代码，也不会漏掉某个入口。
watch(selectedId, (kbId) => {
  chatError.value = null
  ragError.value = null
  ragResult.value = null
  uploadError.value = null
  docsError.value = null
  documents.value = []

  if (!kbId) return
  ensureSession(kbId)
  void loadDocuments(kbId)
})

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

    // 清理本地记录：后端的数据已经没了，留着只会在重新选中同 ID 时
    // 显示一堆实际上已被删除的内容；而会话 ID 更是不能留 ——
    // 它指向的记录已随知识库级联删除，再拿去请求会得到 404。
    const { [knowledgeBase.id]: _droppedSessions, ...restSessions } = sessionsByKb.value
    sessionsByKb.value = restSessions
    const { [knowledgeBase.id]: _droppedActive, ...restActive } = activeSessionByKb.value
    activeSessionByKb.value = restActive
    const { [knowledgeBase.id]: _droppedDocIds, ...restDocIds } = documentIdsByKb.value
    documentIdsByKb.value = restDocIds

    // 置空选中项会触发上面的 watch，把文档、RAG 的界面状态一并清掉。
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

// ---- 文档 ----

/** 把一个文档 ID 从本地记录里去掉（它已在服务端被删除）。 */
function removeDocumentId(id: string): void {
  const next: Record<string, string[]> = {}
  for (const [kbId, ids] of Object.entries(documentIdsByKb.value)) {
    next[kbId] = ids.filter((value) => value !== id)
  }
  documentIdsByKb.value = next
}

/**
 * 按 ID 查一篇文档的当前状态。
 *
 * 404 说明这篇文档已经在别处被删掉了（比如另一个浏览器里删过），
 * 这时把本地记录里的 ID 也一并去掉 —— 否则列表会永远挂着一个查不到的条目，
 * 每次刷新都报一次错。
 */
async function loadOneDocument(id: string): Promise<KnowledgeDocument | null> {
  try {
    return await getDocument(id)
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      removeDocumentId(id)
      return null
    }
    throw error
  }
}

/**
 * 加载当前知识库的文档列表。
 *
 * 这里没有「一次拿到全部文档」的接口可用，只能拿本地记下的 ID 逐个查
 * （后端只提供 GET /api/documents/{id}）。所以这一步的请求数 = 本机上传过的文档数，
 * 文档很多时会有明显的等待 —— 界面上的 loading 状态就是为它准备的。
 */
async function loadDocuments(kbId: string): Promise<void> {
  const ids = documentIdsByKb.value[kbId] ?? []
  docsLoading.value = true
  docsError.value = null
  try {
    const items = await Promise.all(ids.map((id) => loadOneDocument(id)))

    // 请求发出后用户可能已经切到别的知识库了，这时不能再把结果写进去，
    // 否则新库的列表里会闪出上一个库的文档。
    if (selectedId.value !== kbId) return
    documents.value = items.filter((doc): doc is KnowledgeDocument => doc !== null)
  } catch (error) {
    if (selectedId.value !== kbId) return
    docsError.value = toUserMessage(error)
  } finally {
    if (selectedId.value === kbId) {
      docsLoading.value = false
    }
  }
}

function handleRefreshDocuments(): void {
  const kbId = selectedId.value
  if (!kbId) return
  void loadDocuments(kbId)
}

async function handleUpload(file: File): Promise<void> {
  const kbId = selectedId.value
  // uploading 时直接返回：上传接口是同步入库的，重复提交会把同一份文件入库两遍，
  // 而按钮虽然在界面上被禁用了，这道判断仍是最后一道防线。
  if (!kbId || uploading.value) return

  uploading.value = true
  uploadError.value = null
  try {
    const created = await uploadDocument(file, kbId)

    // 上传接口返回的就是入库完成后的文档（status 已是 completed），
    // 直接放进列表，不用再查一次。最新的排在最前面。
    documentIdsByKb.value = {
      ...documentIdsByKb.value,
      [kbId]: [created.id, ...(documentIdsByKb.value[kbId] ?? [])],
    }
    if (selectedId.value === kbId) {
      documents.value = [created, ...documents.value]
    }

    // 上传会改变知识库的文档数量，刷新列表让左侧的计数跟着更新。
    await loadKnowledgeBases()
  } catch (error) {
    // 失败时【不把它加进本地列表】：上传接口失败时不会返回文档 ID，
    // 前端拿不到它，也就没法在之后查到它的状态。这份失败的文档在后端是存在的
    // （status=failed，可以在服务端日志/数据库里看到），这里如实告知用户。
    uploadError.value = toUserMessage(error)
  } finally {
    uploading.value = false
  }
}

async function handleDeleteDocument(document: KnowledgeDocument): Promise<void> {
  const confirmed = window.confirm(
    `确定删除文档「${document.name}」吗？\n` +
      `它在向量库里的切片、数据库记录和原始文件都会被删除，且无法恢复。`,
  )
  if (!confirmed) return

  deletingDocId.value = document.id
  try {
    await deleteDocument(document.id)
    removeDocumentId(document.id)
    documents.value = documents.value.filter((item) => item.id !== document.id)
    // 文档少了，知识库的计数也要跟着更新。
    await loadKnowledgeBases()
  } catch (error) {
    docsError.value = toUserMessage(error)
  } finally {
    deletingDocId.value = null
  }
}

// ---- RAG ----

async function handleAsk(question: string, topK: number): Promise<void> {
  const kbId = selectedId.value
  if (!kbId || ragAsking.value) return

  ragAsking.value = true
  ragError.value = null
  try {
    const answer = await askKnowledgeBase(kbId, { question, top_k: topK })
    // 同 loadDocuments：请求期间用户可能已经切库，结果不能再往界面上写。
    if (selectedId.value !== kbId) return
    ragResult.value = answer
  } catch (error) {
    if (selectedId.value !== kbId) return
    ragError.value = toUserMessage(error)
  } finally {
    ragAsking.value = false
  }
}

// ---- Agent ----

/**
 * 向 Agent 提问。
 *
 * 流程：先把用户消息放进会话（立刻可见），再去请求；拿回答案后再追加
 * assistant 消息。失败时【保留用户消息】—— 那是他确实说过的话，
 * 抹掉会让人怀疑自己到底发出去没有。
 */
async function handleSend(text: string): Promise<void> {
  const kbId = selectedId.value
  const session = activeSession.value
  // thinking 时直接返回：输入框虽然已经禁用了，但键盘操作或将来
  // 加入别的触发入口时，这道判断是最后一道防线。
  if (!kbId || !session || thinking.value) return

  // 把 kbId 和 localId 先记下来：请求返回时用户可能已经切到别的会话/知识库，
  // 那时答案仍然应该落到【提问时】的那个会话里，而不是当前显示的那个。
  const localId = session.localId

  updateSession(kbId, localId, (current) => ({
    ...current,
    // 第一句话顺手拿来当标题，和后端用首句做标题的做法一致。
    title:
      current.messages.length === 0 && current.title === '新会话'
        ? text.slice(0, TITLE_MAX_LENGTH)
        : current.title,
    messages: [
      ...current.messages,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: text,
        createdAt: new Date().toISOString(),
      },
    ],
  }))

  thinking.value = true
  chatError.value = null

  try {
    const response = await askAgent(kbId, {
      question: text,
      // 这个会话已经有后端 ID 就带上，没有（第一次提问）就不带 ——
      // 后端会新建一个并把 ID 返回。带上 null 和不带是等价的，这里显式转成
      // undefined 是为了让 JSON 里干脆不出现这个键。
      conversation_id: session.conversationId ?? undefined,
    })

    updateSession(kbId, localId, (current) => ({
      ...current,
      // 先记下会话 ID：即使下面追加消息时出错，会话本身也是有效的，
      // 丢掉它会让下一轮提问又开一个新会话，上下文白丢。
      conversationId: response.conversation_id,
      messages: [
        ...current.messages,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer,
          createdAt: new Date().toISOString(),
          // 把这一轮的工具调用和运行记录挂在回答上，
          // 让用户能看到答案是怎么来的、慢在哪。
          toolCalls: response.tool_calls,
          trace: {
            trace_id: response.trace_id,
            iterations: response.iterations,
            llm_calls: response.llm_calls,
            total_duration_ms: response.total_duration_ms,
          },
        },
      ],
    }))
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
      <!-- 三个功能区用 v-show 而不是 v-if：v-if 会销毁组件，
           切回来时 RAG 的「检索条数」会被重置、对话的滚动位置也会丢，
           而这些都不该因为切了个标签页就忘掉。 -->
      <nav class="tabs" role="tablist" aria-label="功能切换">
        <button
          v-for="tab in TABS"
          :key="tab.key"
          class="tabs__item"
          :class="{ 'tabs__item--active': activeTab === tab.key }"
          type="button"
          role="tab"
          :aria-selected="activeTab === tab.key"
          @click="activeTab = tab.key"
        >
          {{ tab.label }}
        </button>
      </nav>

      <div class="layout__panel">
        <DocumentPanel
          v-show="activeTab === 'documents'"
          :knowledge-base="selectedKnowledgeBase"
          :documents="documents"
          :loading="docsLoading"
          :error="docsError"
          :uploading="uploading"
          :upload-error="uploadError"
          :deleting-id="deletingDocId"
          @upload="handleUpload"
          @delete="handleDeleteDocument"
          @refresh="handleRefreshDocuments"
        />

        <RagPanel
          v-show="activeTab === 'rag'"
          :knowledge-base="selectedKnowledgeBase"
          :asking="ragAsking"
          :error="ragError"
          :result="ragResult"
          @ask="handleAsk"
        />

        <ChatPanel
          v-show="activeTab === 'agent'"
          :knowledge-base="selectedKnowledgeBase"
          :messages="currentMessages"
          :sessions="currentSessions"
          :active-session-id="activeSessionId"
          :thinking="thinking"
          :error="chatError"
          @send="handleSend"
          @new-session="handleNewSession"
          @select-session="handleSelectSession"
        />
      </div>
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
  display: flex;
  flex-direction: column;
  flex: 1;
  /* min-width: 0 是这条布局里最关键的一行。
     flex 子项默认 min-width: auto，意思是「不能比内容的自然宽度更窄」，
     于是窗口变窄时右侧内容宁可把整体撑出横向滚动条也不肯收缩。
     显式设成 0 之后它才会跟着窗口变窄，左侧栏也就不会被挤变形。 */
  min-width: 0;
}

.tabs {
  display: flex;
  flex-shrink: 0;
  gap: 4px;
  padding: 0 16px;
  background-color: #fff;
  border-bottom: 1px solid var(--color-border);
}

.tabs__item {
  padding: 12px 14px;
  font: inherit;
  font-size: 14px;
  color: var(--color-text-muted);
  background-color: transparent;
  border: none;
  /* 底部留一条透明边：选中时它变成强调色。
     始终占位是为了让切换标签时文字不上下跳动。 */
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}

.tabs__item:hover {
  color: var(--color-text);
}

.tabs__item--active {
  font-weight: 500;
  color: var(--color-accent-strong);
  border-bottom-color: var(--color-accent);
}

.layout__panel {
  flex: 1;
  /* min-height: 0 让内部的可滚动区域真的能滚动：
     默认 min-height: auto 时内容会把容器撑高，滚动条就不出现了。 */
  min-height: 0;
}
</style>
