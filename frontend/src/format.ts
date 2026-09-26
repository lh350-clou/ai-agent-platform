/**
 * 展示层的格式化函数。
 *
 * 集中在一个文件里，是因为同一份数据会在好几处显示（耗时在 Trace 面板、
 * 时间在文档列表、相似度在 RAG 来源里），各写一份的话同一种数据会呈现出
 * 不同的样子 —— 比如一处显示 "1820.4ms"、另一处显示 "1.8 s"。
 */

/**
 * 把后端返回的 ISO 时间字符串格式化成本地时间。
 *
 * 后端返回的是带时区的 UTC 字符串，`new Date()` 解析后会自动转到浏览器所在时区，
 * 所以这里不需要手动做时区换算。
 */
export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  // 时间字符串可能是空串或格式异常（比如本地存下的旧数据），
  // 那种情况下 toLocaleString 会输出 "Invalid Date"，不如直接给个占位符。
  if (Number.isNaN(date.getTime())) return '—'

  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** 把毫秒数格式化成好读的耗时：1000 以下用毫秒，以上用秒。 */
export function formatDurationMs(milliseconds: number): string {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`
  return `${(milliseconds / 1000).toFixed(2)} s`
}

/**
 * 格式化余弦相似度。
 *
 * 保留 4 位小数：这个值通常在 0.3~0.9 之间，只保留 2 位的话
 * 相邻两条结果经常显示成同一个数，反而看不出差距。
 */
export function formatScore(score: number): string {
  return score.toFixed(4)
}
