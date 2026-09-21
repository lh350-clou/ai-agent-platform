/**
 * 前端共享类型。
 *
 * 字段名与后端接口【完全一致】（包括 snake_case），刻意不做 camelCase 转换：
 * 多一层字段名映射就多一处可能写错的地方，而写错的后果是字段静默变成
 * undefined —— 页面不报错，只是显示空白，很难查。
 * 保持一致之后，后端的响应可以直接当这些类型用。
 *
 * 类型放在一个文件里，是因为 KnowledgeBase 和 ChatMessage 会被多个组件
 * 同时用到，各写一份迟早会漂移。
 */

/** 知识库。对应后端 KnowledgeBaseResponse。 */
export interface KnowledgeBase {
  id: string
  name: string
  description: string | null
  /** ISO 8601 字符串，渲染时再格式化。 */
  created_at: string
  updated_at: string
  /** 文档数量，由后端实时统计（不是存在表里的计数字段）。 */
  document_count: number
}

/** 创建知识库的请求体。对应后端 KnowledgeBaseCreate。 */
export interface KnowledgeBaseCreateRequest {
  name: string
  description?: string | null
}

/**
 * 消息角色。
 *
 * 用字面量联合而不是 enum：运行时就是普通字符串，与后端返回的
 * role 字段直接对得上。好处是拼错（"userr"）会被 TS 当场拦下。
 */
export type ChatRole = 'user' | 'assistant'

/** Agent 请求体。对应后端 AgentRequest。 */
export interface AgentRequest {
  question: string
  /**
   * 会话 ID。不传表示开一个新会话，后端创建后随响应返回；
   * 传上一次的返回值则延续同一个会话，模型能看到之前的问答。
   */
  conversation_id?: string
}

/**
 * 一次实际执行过的工具调用。对应后端 AgentToolCall。
 *
 * 只描述「后端告诉我们它执行了什么」，不包含任何内部信息 ——
 * 后端返回的也只有工具名和参数，没有异常堆栈、地址之类的东西。
 */
export interface AgentToolCall {
  tool: string
  /** 本次调用实际使用的参数，键取决于具体工具（检索是 query/top_k，时间工具是 timezone）。 */
  arguments: Record<string, unknown>
  /** 检索工具用的便捷字段；非检索工具为 null。 */
  query: string | null
  top_k: number | null
}

/** Agent 响应体。对应后端 AgentResponse。 */
export interface AgentResponse {
  /** 本次问答所属的会话 ID。存下来，下一轮原样传回去就能接上上下文。 */
  conversation_id: string
  answer: string
  tool_calls: AgentToolCall[]
}

/** 界面上的一条对话消息。 */
export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  createdAt: string
  /** 仅 assistant 消息可能有：这一轮回答过程中调用过哪些工具。 */
  toolCalls?: AgentToolCall[]
}
