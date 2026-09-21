/** Agent 对话接口。对应后端 app/api/agent.py。 */

import { apiRequest } from './client'
import type { AgentRequest, AgentResponse } from '../types'

/**
 * 向指定知识库的 Agent 提问。
 *
 * 知识库 ID 放在路径里而不是请求体里，和后端的设计一致：
 * 检索范围由服务端根据 URL 决定，模型和客户端都改不了，
 * 这样「跨库检索」在接口层面就无法表达。
 *
 * 注意这是一个【慢接口】：后端要先向量化问题、检索 Milvus，
 * 再（可能多轮地）调用 DeepSeek，一次几秒到十几秒都正常。
 * 调用方必须自己处理等待状态，不要因为「没立刻返回」就重试 ——
 * 重试会让后端多跑一遍完整流程。
 */
export function askAgent(
  knowledgeBaseId: string,
  payload: AgentRequest,
): Promise<AgentResponse> {
  return apiRequest<AgentResponse>(
    `/api/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/agent`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
  )
}
