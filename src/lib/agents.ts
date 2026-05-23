// Клиент Python-сервера D&D Core AI.
// Новый рекомендуемый flow — звать /game/turn (один RPC, весь pipeline на сервере).
// Старые helper'ы (callLocalAgent, runAgent) сохранены для совместимости.

// Авто-определение адреса Python-сервера.
// Логика:
// 1) Если задано VITE_PYTHON_SERVER в .env — берём его (явный override).
// 2) Если страница открыта в браузере — берём хост из window.location и порт 8000.
//    Это работает и для localhost, и для LAN (192.168.x.x), и для туннелей (ngrok/cloudflare).
// 3) Фоллбэк для SSR/Node — localhost.
function resolveServerUrl(): string {
  const envUrl =
    (typeof process !== "undefined" && process.env && process.env.VITE_PYTHON_SERVER) ||
    (import.meta as any).env?.VITE_PYTHON_SERVER;
  if (envUrl) return envUrl;

  if (typeof window !== "undefined" && window.location) {
    // Берём хост из URL, который пользователь открыл, и подменяем порт на 8000.
    // Пример: открыт http://192.168.1.5:5173/ → API будет http://192.168.1.5:8000
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return "http://localhost:8000";
}

const PYTHON_SERVER = resolveServerUrl();

// Полный ход = 5 LLM-вызовов последовательно.
// - GPU + квантование: 5-15 сек
// - GPU без квантования (AMD DirectML): 15-40 сек
// - CPU fallback: 60-180 сек (когда GPU не подхватился)
// Ставим 240s чтобы фронт не отваливался на CPU — это удобно для отладки.
// Если на GPU всё работает быстро, ничего не теряем (таймаут — это потолок, не задержка).
const TURN_TIMEOUT_MS = 240_000;
const SINGLE_AGENT_TIMEOUT_MS = 60_000;

export interface TurnResult {
  status: "success" | "error";
  time: number;
  category: string;
  intent: Record<string, any>;
  mechanics: {
    success: boolean;
    roll: number;
    total: number;
    dc: number;
    damage: number;
    critical: "hit" | "miss" | null;
    narration_hint: string;
  };
  tactic: Record<string, any> | null;
  narration: string;
  keeper_update: Record<string, any>;
  world_state: Record<string, any>;
}

export interface TurnHistoryEntry {
  player: string;
  master: string;
}

export async function playTurn(
  userInput: string,
  worldState: Record<string, any>,
  history: TurnHistoryEntry[] = []
): Promise<TurnResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TURN_TIMEOUT_MS);
  try {
    const res = await fetch(`${PYTHON_SERVER}/game/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_input: userInput,
        world_state: worldState,
        history: history.slice(-3), // 3 последних хода — больше не нужно, контекст забьётся
      }),
      signal: controller.signal as any,
    });
    if (!res.ok) {
      throw new Error(`Turn failed: ${res.status} ${res.statusText}`);
    }
    return (await res.json()) as TurnResult;
  } finally {
    clearTimeout(timeoutId);
  }
}

// === Legacy single-agent helpers (для отладки/обратной совместимости) ===

interface AgentResponse {
  response: string;
  time: number;
}

export async function callLocalAgent(
  mode: string,
  prompt: string,
  context: string = ""
): Promise<string> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), SINGLE_AGENT_TIMEOUT_MS);

    const response = await fetch(`${PYTHON_SERVER}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, prompt, context }),
      signal: controller.signal as any,
    });
    clearTimeout(timeoutId);
    if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
    const data: AgentResponse = await response.json();
    console.log(`[${mode}] ${data.time}s`);
    return data.response;
  } catch (error) {
    console.error(`Agent error (${mode}):`, error);
    return "";
  }
}

export async function callAgentsParallel(
  agents: Array<{ mode: string; prompt: string; context?: string }>
): Promise<string[]> {
  return Promise.all(agents.map((a) => callLocalAgent(a.mode, a.prompt, a.context || "")));
}

export const AGENT_PROMPTS = {
  ORCHESTRATOR: "АТАКА/НАВЫК/ЗАКЛИНАНИЕ/ДИАЛОГ/ИССЛЕДОВАНИЕ/НЕВОЗМОЖНО",
  JUDGE: "JSON-intent для DnDEngine",
  STORYTELLER: "Атмосферное описание от третьего лица",
  KEEPER: "JSON state delta",
  TACTICIAN: "Действие монстра",
};

export async function runAgent(
  agentKey: keyof typeof AGENT_PROMPTS,
  input: string,
  context: any = {}
): Promise<string> {
  const modeMap: Record<string, string> = {
    ORCHESTRATOR: "orchestrator",
    JUDGE: "judge",
    KEEPER: "keeper",
    TACTICIAN: "tactician",
    STORYTELLER: "storyteller",
  };
  const mode = modeMap[agentKey] || agentKey.toLowerCase();
  const contextStr = typeof context === "string" ? context : JSON.stringify(context);
  return callLocalAgent(mode, input, contextStr);
}
