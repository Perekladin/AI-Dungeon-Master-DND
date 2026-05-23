import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";

// Параметры подключения к Python серверу
const PYTHON_SERVER = process.env.PYTHON_SERVER || "http://localhost:8000";
const TIMEOUT_MS = 12000; // 12 секунд максимум

async function callPythonServer(endpoint: string, data: any): Promise<any> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
    
    const response = await fetch(`${PYTHON_SERVER}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
      signal: controller.signal as any,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Python server error: ${error}`);
    throw error;
  }
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Game state (In-memory for this demo)
  let worldState = {
    player: { name: "Warrior", hp: 30, ac: 16, inventory: ["Sword", "Shield"] },
    enemies: [{ name: "Skeleton", hp: 13, ac: 13 }],
    location: "Damp Dungeon Cell",
    history: [] as any[],
  };

  // ОПТИМИЗИРОВАННЫЙ ЭНДПОИНТ: Один вызов вместо 4 последовательных
  app.post("/api/game/turn", async (req, res) => {
    const startTime = Date.now();
    try {
      const { userInput } = req.body;
      
      if (!userInput || userInput.trim().length === 0) {
        return res.status(400).json({ error: "User input is required" });
      }

      // Отправляем запрос на Python сервер - ВСЕ АГЕНТЫ РАБОТАЮТ ПАРАЛЛЕЛЬНО
      const result = await callPythonServer("/game/turn", {
        user_input: userInput.substring(0, 100),
        world_state: worldState
      });

      if (result.time > 12) {
        console.warn(`[!] Turn took ${result.time}s - exceeds 12s limit`);
      }

      // Парсим результаты
      const orchestrator = result.orchestrator || "";
      const judge = result.judge || "";
      const storyteller = result.storyteller || "";
      const keeper = result.keeper || "";

      // Определяем категорию действия
      const category = orchestrator.toUpperCase().includes("АТАКА") ? "ATTACK" :
                      orchestrator.toUpperCase().includes("ДИАЛОГ") ? "DIALOGUE" :
                      orchestrator.toUpperCase().includes("ИССЛЕДОВ") ? "EXPLORE" : "ACTION";

      // Парсим урон из ответа Judge (если есть число)
      const damageMatch = judge.match(/урон:?\s*(\d+)/i);
      const damage = damageMatch ? parseInt(damageMatch[1]) : 0;

      // Обновляем состояние врагов
      if (damage > 0 && worldState.enemies.length > 0) {
        worldState.enemies[0].hp -= damage;
        console.log(`[*] Damage dealt: ${damage}, Enemy HP: ${worldState.enemies[0].hp}`);
        
        if (worldState.enemies[0].hp <= 0) {
          worldState.enemies.shift();
          console.log("[!] Enemy defeated!");
        }
      }

      // История действий
      worldState.history.push({ 
        user: userInput, 
        narrative: storyteller, 
        logic: judge,
        timestamp: new Date().toISOString()
      });
      if (worldState.history.length > 10) worldState.history.shift();

      const totalTime = Date.now() - startTime;
      console.log(`[✓] API turn completed in ${totalTime}ms`);

      res.json({
        category,
        orchestrator,
        judge,
        storyteller,
        keeper,
        damage,
        worldState,
        timing: {
          python_server: result.time,
          total_api: Math.round(totalTime / 1000 * 100) / 100
        }
      });
    } catch (error: any) {
      const elapsed = Date.now() - startTime;
      console.error(`[X] Turn failed after ${elapsed}ms: ${error.message}`);
      
      if (error.name === "AbortError") {
        return res.status(504).json({ error: "Request timeout - exceeded 12 seconds" });
      }
      
      res.status(500).json({ error: error.message || "Agent failure" });
    }
  });

  // Проверка здоровья
  app.get("/api/health", async (req, res) => {
    try {
      const pythonHealth = await fetch(`${PYTHON_SERVER}/health`).then(r => r.json());
      res.json({
        status: "ok",
        server: "running",
        python: pythonHealth,
        world_state: worldState
      });
    } catch {
      res.json({
        status: "warning",
        server: "running",
        python: "offline",
        world_state: worldState
      });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n[✓] Express server running on http://localhost:${PORT}`);
    console.log(`[✓] Python AI server expected at ${PYTHON_SERVER}`);
    console.log(`[✓] Timeout per turn: ${TIMEOUT_MS}ms (12s)\n`);
  });
}

startServer().catch(console.error);

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
