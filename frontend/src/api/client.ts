/**
 * HTTP 客户端：整个前端只有这里直接调用 fetch。
 *
 * 把请求收口到一个文件，是为了让「怎么发请求」这件事只有一处实现 ——
 * base URL、JSON 序列化、错误转换都在这里定死。散在各组件里的话，
 * 每个组件都要自己判断 response.ok、自己解析错误，迟早出现
 * 「有的地方把 500 当成功处理」这类问题。
 */

/**
 * 后端地址。
 *
 * 从环境变量读，不写死在代码里：开发、联调、部署三种场景下后端地址都不同，
 * 而前端产物是构建时确定的，写死就意味着每次换环境都要改代码重新构建。
 *
 * `??` 而不是 `||`：|| 会把空字符串也当成「没配」，
 * 而空字符串在这里是无效值，用 ?? 只兜住 undefined。
 */
const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

/** 带 HTTP 状态码的接口错误，方便调用方按状态码做区分处理。 */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    // name 要显式设置：继承自 Error 时它默认还是 "Error"，
    // 控制台里看不出是哪类错误。
    this.name = 'ApiError'
    this.status = status
  }
}

/** FastAPI 的错误响应体形如 {"detail": "..."} 或 {"detail": [{...校验详情...}]}。 */
interface ErrorBody {
  detail?: unknown
}

/**
 * 从错误响应里抠出一句能给用户看的话。
 *
 * FastAPI 有两种 detail：
 *   - 字符串   —— 业务代码主动抛的 HTTPException，例如「知识库不存在」，可以直接展示；
 *   - 数组     —— Pydantic 校验失败，每项是一个字段错误，取第一条的 msg。
 * 拿不到就返回 null，由调用方兜底成通用文案。
 */
function extractDetail(body: unknown): string | null {
  if (typeof body !== 'object' || body === null) return null

  const { detail } = body as ErrorBody
  if (typeof detail === 'string') return detail

  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: unknown }
    if (typeof first.msg === 'string') return first.msg
  }

  return null
}

/**
 * 发一个请求并返回解析后的 JSON。
 *
 * 参数：
 *   path: 以 / 开头的接口路径，例如 "/api/knowledge-bases"。
 *   init: 透传给 fetch 的配置（method / body 等）。
 *
 * 返回：
 *   解析后的响应体。204 无内容时返回 undefined。
 *
 * 异常：
 *   一律抛 ApiError，消息是【可以直接展示给用户】的中文。
 *   status 为 0 表示请求根本没发出去（网络问题、后端没启动）。
 */
export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    })
  } catch {
    // fetch 只在「请求根本没发出去」时才 reject（后端没启动、断网、被 CORS 拦下）。
    // 这里不把原始异常抛给界面 —— 它长这样："Failed to fetch"，
    // 对用户没有任何帮助，反而会让人以为是业务错误。
    throw new ApiError('无法连接后端服务，请确认后端已启动', 0)
  }

  if (!response.ok) {
    // 先试着读错误详情，读不到（响应体不是 JSON）就退回通用文案。
    let detail: string | null = null
    try {
      detail = extractDetail(await response.json())
    } catch {
      // 响应体不是合法 JSON，忽略，用下面的兜底文案。
    }
    throw new ApiError(detail ?? `请求失败（HTTP ${response.status}）`, response.status)
  }

  // 204 No Content 没有响应体，直接调 response.json() 会抛解析错误。
  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}
