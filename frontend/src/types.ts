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
 *
 * 注意这里【没有】success / duration 字段：后端确实在服务端的 Trace 里
 * 记录了每次工具调用的成败与耗时，但没有把它们暴露到这个结构上
 * （见 backend/app/schemas/agent.py）。所以界面上不能显示「工具调用成功/失败」，
 * 只能显示调用了什么 —— 凭空补一个状态出来就是编数据。
 */
export interface AgentToolCall {
  tool: string
  /** 本次调用实际使用的参数，键取决于具体工具（检索是 query/top_k，时间工具是 timezone）。 */
  arguments: Record<string, unknown>
  /** 检索工具用的便捷字段；非检索工具为 null。 */
  query: string | null
  top_k: number | null
}

/** 一次 LLM 调用。对应后端 AgentLLMCall。 */
export interface AgentLLMCall {
  model: string
  /** 本次调用耗时（毫秒），含网络等待。 */
  duration_ms: number
  success: boolean
  /** 失败原因；成功时为 null。 */
  error: string | null
}

/**
 * 一轮 Agent Run 的运行记录。对应后端 AgentResponse 里的
 * trace_id / iterations / llm_calls / total_duration_ms 四个字段。
 *
 * 单独抽成一个接口，是因为消息要把它整份存下来（见 ChatMessage.trace）：
 * 写成内联字段的话，聊天记录和接口响应就成了两种结构，存的时候得逐个字段抄。
 */
export interface AgentTrace {
  /** 本次运行的 Trace ID，可在服务端日志里按它捞到同一轮的完整记录。 */
  trace_id: string
  /** 「调模型 → 执行工具」循环了几轮。 */
  iterations: number
  /** 每一次 LLM 调用的耗时与成败，按发生顺序。 */
  llm_calls: AgentLLMCall[]
  /** 整个 Agent Run 的总耗时（毫秒）。 */
  total_duration_ms: number
}

/** Agent 响应体。对应后端 AgentResponse。 */
export interface AgentResponse extends AgentTrace {
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
  /** 仅 assistant 消息可能有：这一轮运行记录（耗时、轮数、模型调用）。 */
  trace?: AgentTrace
}

/**
 * 一个会话。
 *
 * 后端没有「会话列表」接口（见 README 的后续规划），会话 ID 只能从
 * /agent 的响应里拿到，所以这份列表由前端自己维护并存在本机。
 *
 * 两个 ID 分开，是因为它们出现的时机不同：
 *   - localId 由前端生成，「新建会话」时立刻就有，用作列表的 key；
 *   - conversationId 是后端的会话 ID，【第一次提问之后】才拿得到 ——
 *     新建的会话在提问前，后端根本还没有这条记录，此时必须是 null。
 */
export interface ChatSession {
  localId: string
  conversationId: string | null
  /** 会话标题，取第一句话的前几十个字符，方便在列表里认出来。 */
  title: string
  createdAt: string
  /** 这个会话的消息。后端有完整历史，这里存一份是为了切换会话时界面能立刻显示。 */
  messages: ChatMessage[]
}

/** 文档的处理状态。对应后端 DocumentStatus。 */
export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed'

/** 文档。对应后端 DocumentResponse / DocumentUploadResponse。 */
export interface KnowledgeDocument {
  id: string
  knowledge_base_id: string
  /** 原始文件名。 */
  name: string
  /** 文件类型，目前只有 txt。 */
  file_type: string
  status: DocumentStatus
  /** 成功写入向量库的切片数量。 */
  chunk_count: number
  /** 失败原因，仅当 status 为 failed 时有值。 */
  error_message: string | null
  created_at: string
  updated_at: string
}

/** 一条检索结果。对应后端 SearchResultItem。 */
export interface SearchResultItem {
  chunk_id: string
  document_id: string
  content: string
  /** 余弦相似度，越大越相似。 */
  score: number
}

/** RAG 问答请求体。对应后端 AskRequest。 */
export interface AskRequest {
  question: string
  /** 检索多少条资料用于回答，后端范围 1~10。 */
  top_k?: number
  /**
   * 会话 ID。本项目的 RAG 面板不传它 —— 每次提问都是独立的一次检索问答，
   * 传入的话后端会把历史一起交给模型，那是 Agent 面板在做的事。
   */
  conversation_id?: string
}

/** RAG 问答响应体。对应后端 AskResponse。 */
export interface AskResponse {
  conversation_id: string
  /** 回显清洗后的问题。 */
  question: string
  answer: string
  /** 本次回答实际使用的检索结果，按相似度降序；没检索到时是空数组。 */
  sources: SearchResultItem[]
}
