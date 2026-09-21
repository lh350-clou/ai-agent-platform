/** 知识库相关接口。路径与后端 app/api/knowledge_bases.py 一一对应。 */

import { apiRequest } from './client'
import type { KnowledgeBase, KnowledgeBaseCreateRequest } from '../types'

/** 列出全部知识库，按创建时间倒序。没有知识库时返回空数组（不是错误）。 */
export function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  return apiRequest<KnowledgeBase[]>('/api/knowledge-bases')
}

/** 按 ID 查询单个知识库。不存在时抛 ApiError（status 404）。 */
export function getKnowledgeBase(id: string): Promise<KnowledgeBase> {
  // encodeURIComponent 在这里其实是保险：id 是 UUID，不含需要转义的字符。
  // 但地址栏参数一旦将来换成别的形式（比如带用户输入的名字），
  // 少这一步就会让 / 或 ? 把路径结构撑坏。
  return apiRequest<KnowledgeBase>(`/api/knowledge-bases/${encodeURIComponent(id)}`)
}

/** 创建知识库。 */
export function createKnowledgeBase(payload: KnowledgeBaseCreateRequest): Promise<KnowledgeBase> {
  return apiRequest<KnowledgeBase>('/api/knowledge-bases', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * 删除知识库，连同它的文档、会话和向量数据。
 *
 * 返回 void：后端返回 204 无内容。
 * 这是不可逆操作，调用方应当先跟用户确认。
 */
export function deleteKnowledgeBase(id: string): Promise<void> {
  return apiRequest<void>(`/api/knowledge-bases/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
}
