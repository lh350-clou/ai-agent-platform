/** RAG 问答接口。对应后端 app/api/qa.py。 */

import { apiRequest } from './client'
import type { AskRequest, AskResponse } from '../types'

/**
 * 基于知识库内容提问（RAG：先检索、再把命中的资料交给大模型作答）。
 *
 * 和 Agent 的区别：这条路每次都会先检索一遍，再由模型基于检索结果回答，
 * 模型没有「要不要查」的选择权。所以「资料不足时明确说无法确定」是它的正常输出，
 * 而不是错误。
 *
 * 同样是【慢接口】：一次请求会走 embedding + Milvus 检索 + DeepSeek 调用。
 */
export function askKnowledgeBase(
  knowledgeBaseId: string,
  payload: AskRequest,
): Promise<AskResponse> {
  return apiRequest<AskResponse>(
    `/api/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/ask`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
  )
}
