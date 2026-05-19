// Оптимизированные агенты - используют локальный Python сервер вместо Google Gemini API
// Это значительно ускоряет обработку (с 20-30s до 5-12s)

const PYTHON_SERVER = process.env.VITE_PYTHON_SERVER || "http://localhost:8000";
const TIMEOUT_MS = 10000; // Timeout для каждого вызова (10 сек)

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
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const response = await fetch(`${PYTHON_SERVER}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        prompt: prompt.substring(0, 100),
        context: context.substring(0, 200),
      }),
      signal: controller.signal as any,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }

    const data: AgentResponse = await response.json();
    console.log(`[${mode}] Response time: ${data.time}s`);
    return data.response;
  } catch (error) {
    console.error(`Agent error (${mode}):`, error);
    return "";
  }
}

// Быстрые параллельные вызовы для нескольких агентов
export async function callAgentsParallel(
  agents: Array<{ mode: string; prompt: string; context?: string }>
): Promise<string[]> {
  const promises = agents.map((agent) =>
    callLocalAgent(agent.mode, agent.prompt, agent.context || "")
  );
  return Promise.all(promises);
}

// Старые функции для совместимости (теперь используют локальный сервер)
export const AGENT_PROMPTS = {
  ORCHESTRATOR: `Классифицируй действие. Ответь одним словом: АТАКА, ДЕЙСТВИЕ, ДИАЛОГ или ИССЛЕДОВАНИЕ.`,
  JUDGE: `Ты судья D&D 5e. Определи успешность действия: УСПЕХ или ПРОВАЛ. Если урон - напиши число.`,
  STORYTELLER: `Ты мастер подземелий. Опиши атмосферно (1-2 предложения). Используй образный язык.`,
  KEEPER: `Ты хранитель памяти. Кратко опиши: ХП, врагов, локацию. Фактами.`,
};

export async function runAgent(
  agentKey: keyof typeof AGENT_PROMPTS,
  input: string,
  context: any = {}
): Promise<string> {
  // Преобразуем ключи в имена моделей Python сервера
  const modeMap: Record<string, string> = {
    ORCHESTRATOR: "orchestrator",
    JUDGE: "judge",
    KEEPER: "keeper",
    STORYTELLER: "storyteller",
  };

  const mode = modeMap[agentKey] || agentKey.toLowerCase();
  const contextStr = JSON.stringify(context).substring(0, 200);

  return callLocalAgent(mode, input, contextStr);
}
