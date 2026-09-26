/**
 * 前端本地状态（localStorage）。
 *
 * 为什么需要它：后端目前【没有】「文档列表」和「会话列表」这两个接口 ——
 * 文档只能按 ID 查（GET /api/documents/{id}），会话 ID 只能从 /agent 的响应里拿到。
 * 于是「这个知识库下我上传过哪些文档」「我开过哪些会话」这两件事，
 * 前端必须自己记住，否则一刷新页面列表就空了。
 *
 * 存的边界要划清楚：
 *   - 文档只存 ID。名称、状态、切片数在加载时全部重新向后端查一遍，
 *     这里只负责回答「要查哪些 ID」。这样界面上显示的永远是服务端的真实状态，
 *     而不是一份可能已经过期的本地快照。
 *   - 会话要连消息一起存。后端有完整的历史，但没有「按会话读消息」的接口，
 *     不留一份副本的话，切换会话时界面只能是一片空白。
 *
 * 存不上不算错误：隐私模式下 localStorage 会直接抛异常，配额满了也会。
 * 这里一律吞掉并退回空状态 —— 本地记录丢了只是列表少几行，
 * 不该因此让整个页面打不开。
 */

import type { ChatMessage, ChatSession } from './types'

/**
 * 存储键。
 *
 * 带一个版本号后缀：以后结构变了（比如 ChatSession 换字段），
 * 直接改成 v2，旧数据就自然失效、不会被当成新结构解析。
 * 比在读取时写一堆兼容分支简单得多。
 */
const STORAGE_KEY = 'ai-kb-platform:v1'

/** 每个会话本地最多保留多少条消息。 */
// 需要一个上限，是因为 localStorage 通常只有几 MB：对话是只增不减的，
// 攒够几千条长回答就会写不进去（QuotaExceededError），
// 那之后所有本地记录（含文档列表）都会停止更新 —— 坏在了一个看起来无关的地方。
// 100 条远超屏幕上能回看的范围，超出的部分后端仍然有完整历史。
const MAX_MESSAGES_PER_SESSION: number = 100

/** 本地保存的全部状态。 */
export interface LocalState {
  /** 知识库 ID -> 它的会话列表。 */
  sessions: Record<string, ChatSession[]>
  /** 知识库 ID -> 当前选中的会话 localId。 */
  activeSession: Record<string, string>
  /** 知识库 ID -> 本机上传过的文档 ID，最新的在前。 */
  documentIds: Record<string, string[]>
}

/** 空状态。读取失败、没存过、结构不对，都用它兜底。 */
export function emptyLocalState(): LocalState {
  return { sessions: {}, activeSession: {}, documentIds: {} }
}

/** 判断一个值是不是「键值都是数组」的普通对象，用来挡住结构错乱的残留数据。 */
function isArrayMap(value: unknown): value is Record<string, unknown[]> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  return Object.values(value).every((item) => Array.isArray(item))
}

/**
 * 把存下来的一条消息还原成 ChatMessage。
 *
 * 逐字段挑而不是直接类型断言：localStorage 里的内容是可以被手工改的，
 * 也可能是旧版本写下的。少一个字段（比如 content 是 undefined）会让模板渲染出
 * 「undefined」；这里挑不出来就丢掉这条消息，界面最多少一条，不会显示乱码。
 */
function normalizeMessage(raw: unknown): ChatMessage | null {
  if (typeof raw !== 'object' || raw === null) return null
  const message = raw as Partial<ChatMessage>
  if (typeof message.id !== 'string' || typeof message.content !== 'string') return null
  if (message.role !== 'user' && message.role !== 'assistant') return null

  return {
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: typeof message.createdAt === 'string' ? message.createdAt : '',
    toolCalls: Array.isArray(message.toolCalls) ? message.toolCalls : undefined,
    trace: typeof message.trace === 'object' && message.trace !== null ? message.trace : undefined,
  }
}

function normalizeSession(raw: unknown): ChatSession | null {
  if (typeof raw !== 'object' || raw === null) return null
  const session = raw as Partial<ChatSession>
  if (typeof session.localId !== 'string') return null

  const messages = Array.isArray(session.messages)
    ? session.messages
        .map(normalizeMessage)
        .filter((message): message is ChatMessage => message !== null)
    : []

  return {
    localId: session.localId,
    // conversationId 为 null 是合法状态：新建但还没提问过的会话就是没有后端 ID。
    conversationId: typeof session.conversationId === 'string' ? session.conversationId : null,
    title: typeof session.title === 'string' && session.title ? session.title : '新会话',
    createdAt: typeof session.createdAt === 'string' ? session.createdAt : '',
    messages,
  }
}

/** 读取本地状态。任何异常都退回空状态，绝不抛给调用方。 */
export function loadLocalState(): LocalState {
  let raw: string | null = null
  try {
    raw = window.localStorage.getItem(STORAGE_KEY)
  } catch {
    // 隐私模式 / 禁用存储：读不到就当没存过。
    return emptyLocalState()
  }
  if (!raw) return emptyLocalState()

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return emptyLocalState()
  }
  if (typeof parsed !== 'object' || parsed === null) return emptyLocalState()

  const state = parsed as Partial<LocalState>
  const sessions: Record<string, ChatSession[]> = {}
  if (isArrayMap(state.sessions)) {
    for (const [kbId, list] of Object.entries(state.sessions)) {
      sessions[kbId] = list
        .map(normalizeSession)
        .filter((session): session is ChatSession => session !== null)
    }
  }

  const activeSession: Record<string, string> = {}
  if (typeof state.activeSession === 'object' && state.activeSession !== null) {
    for (const [kbId, localId] of Object.entries(state.activeSession)) {
      if (typeof localId === 'string') activeSession[kbId] = localId
    }
  }

  const documentIds: Record<string, string[]> = {}
  if (isArrayMap(state.documentIds)) {
    for (const [kbId, list] of Object.entries(state.documentIds)) {
      documentIds[kbId] = list.filter((id): id is string => typeof id === 'string')
    }
  }

  return { sessions, activeSession, documentIds }
}

/** 写入本地状态。写不进去（配额满、隐私模式）就静默放弃。 */
export function saveLocalState(state: LocalState): void {
  const trimmed: LocalState = {
    sessions: {},
    activeSession: state.activeSession,
    documentIds: state.documentIds,
  }

  // 落盘前才截断消息条数，而不是在内存里截 ——
  // 内存里保留完整的一轮对话，界面滚动回看不会缺内容；
  // 只有写进 localStorage 的那份需要限制大小。
  for (const [kbId, list] of Object.entries(state.sessions)) {
    trimmed.sessions[kbId] = list.map((session) => ({
      ...session,
      messages: session.messages.slice(-MAX_MESSAGES_PER_SESSION),
    }))
  }

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed))
  } catch {
    // 不提示用户：本地记录只影响「列表记不记得住」，
    // 弹一个错误出来反而像是主流程失败了。
  }
}
