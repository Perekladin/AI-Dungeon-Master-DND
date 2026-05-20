// Клиент Python-сервера D&D Core AI.
// Новый рекомендуемый flow — звать /game/turn (один RPC, весь pipeline на сервере).
// Старые helper'ы (callLocalAgent, runAgent) сохранены для совместимости.

const PYTHON_SERVER =
  (typeof process !== "undefined" && process.env && process.env.VITE_PYTHON_SERVER) ||
  (import.meta as any).env?.VITE_PYTHON_SERVER ||
  "http://localhost:8000";

// Один полный ход может занять до 25 секунд на холодном AMD без квантования.
const TURN_TIMEOUT_MS = 30_000;
const SINGLE_AGENT_TIMEOUT_MS = 12_000;

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

export async function playTurn(
  userInput: string,
  worldState: Record<string, any>
): Promise<TurnResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TURN_TIMEOUT_MS);
  try {
    const res = await fetch(`${PYTHON_SERVER}/game/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input: userInput, world_state: worldState }),
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
