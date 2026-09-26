/** 文档接口。路径与后端 app/api/documents.py 一一对应。 */

import { apiRequest } from './client'
import type { KnowledgeDocument } from '../types'

/**
 * 上传一份 TXT 文档并入库。
 *
 * 后端要的是 multipart/form-data，两个字段名（file、knowledge_base_id）
 * 必须与后端 upload_document 的形参名一模一样 —— 对不上的话 FastAPI
 * 会以 422 报「字段缺失」，而它并不知道我们想传的是什么。
 *
 * 注意这是一个【慢接口】：后端是同步入库的（解析 → 切分 → 向量化 → 写 Milvus
 * 全在这一次请求里做完），大文件要等几十秒。所以调用方必须自己显示上传中状态，
 * 也不能因为「没立刻返回」就重试 —— 重试会让后端把同一份文件入库两遍。
 */
export function uploadDocument(
  file: File,
  knowledgeBaseId: string,
): Promise<KnowledgeDocument> {
  const form = new FormData()
  form.append('file', file)
  form.append('knowledge_base_id', knowledgeBaseId)

  return apiRequest<KnowledgeDocument>('/api/documents/upload', {
    method: 'POST',
    body: form,
  })
}

/** 按 ID 查询文档的处理状态。不存在时抛 ApiError（status 404）。 */
export function getDocument(id: string): Promise<KnowledgeDocument> {
  return apiRequest<KnowledgeDocument>(`/api/documents/${encodeURIComponent(id)}`)
}

/**
 * 删除一份文档：后端会清理它的 Milvus 向量、PostgreSQL 记录和磁盘文件。
 *
 * 返回 void：后端返回 204 无内容。
 * 这是不可逆操作，调用方应当先跟用户确认。
 */
export function deleteDocument(id: string): Promise<void> {
  return apiRequest<void>(`/api/documents/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
}
