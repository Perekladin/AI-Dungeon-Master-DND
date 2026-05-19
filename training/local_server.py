import os
import sys
import socket
import torch
import gc
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dnd_engine import DnDEngine

app = FastAPI(title="D&D Core AI High-Speed Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пул потоков для параллельной обработки
executor = ThreadPoolExecutor(max_workers=4)

PYTHON_SERVER_HOST = os.environ.get("PYTHON_SERVER_HOST", "0.0.0.0")
PYTHON_SERVER_PORT = int(os.environ.get("PYTHON_SERVER_PORT", 8000))


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(start_port: int = 8000, end_port: int = 8100) -> int:
    for port in range(start_port, end_port + 1):
        if is_port_free(PYTHON_SERVER_HOST, port):
            return port
    raise RuntimeError(f"No available port found between {start_port} and {end_port}")


def has_directml() -> bool:
    try:
        import torch_directml
        return True
    except Exception:
        return False


def has_rocm() -> bool:
    try:
        # torch.version.hip is present on ROCm builds
        return getattr(torch.version, "hip", None) is not None or "rocm" in (getattr(torch.version, "__version__", "").lower())
    except Exception:
        return False


def get_torch_device(device_name: str):
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "mps":
        return torch.device("mps")
    if device_name == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


def get_device():
    global ROCM_ENABLED
    if has_rocm():
        ROCM_ENABLED = True
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if has_directml():
        return "dml"
    return "cpu"

ROCM_ENABLED = False
DEVICE = get_device()
TORCH_DEVICE = get_torch_device(DEVICE)
MODEL_CACHE = {}

# Исправленный путь - относительно расположения скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTERS_BASE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "dnd_core_models")

MODEL_MAP = {
    "orchestrator": "Qwen/Qwen2.5-0.5B-Instruct",
    "judge": "Qwen/Qwen2.5-1.5B-Instruct",
    "keeper": "Qwen/Qwen2.5-1.5B-Instruct",
    "storyteller": "Qwen/Qwen2.5-3B-Instruct"
}

# Быстрые параметры для каждого агента
AGENT_CONFIG = {
    "orchestrator": {
        "max_tokens": 50,
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.2
    },
    "judge": {
        "max_tokens": 100,
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.2
    },
    "keeper": {
        "max_tokens": 80,
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.2
    },
    "storyteller": {
        "max_tokens": 200,
        "do_sample": True,
        "temperature": 0.85,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.3
    }
}

def load_model(mode: str):
    if mode in MODEL_CACHE:
        return MODEL_CACHE[mode]
    
    base_id = MODEL_MAP.get(mode)
    adapter_path = os.path.join(ADAPTERS_BASE_PATH, f"{mode}_lora_adapter")
    
    print(f"[*] Pre-loading {mode}...")
    print(f"    Base model: {base_id}")
    print(f"    Adapter path: {adapter_path}")
    
    # Проверяем что адаптер существует
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at: {adapter_path}")
    
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config}")
    
    quant_config = None
    # bitsandbytes quantization is CUDA-only; avoid enabling it for ROCm/DirectML
    if DEVICE == "cuda" and not ROCM_ENABLED:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    model_kwargs = {
        "device_map": "auto" if (DEVICE == "cuda" or DEVICE == "rocm") else None,
        "torch_dtype": torch.float16 if DEVICE in ("cuda", "rocm") else torch.float32,
        "low_cpu_mem_usage": True
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    try:
        base = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)
        if DEVICE != "cuda":
            base = base.to(TORCH_DEVICE)
        
        model = PeftModel.from_pretrained(base, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Warmup
        _ = model.generate(**tokenizer("Warmup", return_tensors="pt").to(TORCH_DEVICE), max_new_tokens=1)
        
        MODEL_CACHE[mode] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"[X] Failed to load {mode}: {str(e)}")
        raise

# Принудительная загрузка всех моделей при старте (если памяти хватит)
@app.on_event("startup")
async def startup_event():
    print("[!] Warming up AI engines... Please wait.")
    print(f"[i] Device detected: {DEVICE}")
    print(f"[i] Torch device: {TORCH_DEVICE}")
    print(f"[i] PyTorch version: {torch.__version__}")
    if DEVICE == "cpu":
        print("[!] Warning: GPU device not detected. If у тебя AMD, установи ROCm или torch-directml.")
        print("[!] Для Windows AMD попробуй: pip install --pre torch-directml -f https://raw.githubusercontent.com/microsoft/DirectML/main/packaging/wheels/directml.html")
    if DEVICE == "rocm":
        print("[i] ROCm detected. Models will load targeting ROCm backend (torch.device('cuda')).")
    for mode in ["orchestrator", "judge", "storyteller", "keeper"]:
        try:
            load_model(mode)
            print(f"[+] {mode.capitalize()} loaded successfully")
        except Exception as e:
            print(f"[⚠] Warning loading {mode}: {str(e)[:100]}")
            print(f"[*] Will load on-demand when needed")

class ChatRequest(BaseModel):
    mode: str
    prompt: str
    context: str = ""

class ParallelRequest(BaseModel):
    user_input: str
    world_state: dict = {}

def generate_fast(mode: str, prompt: str, context: str = "") -> str:
    """Быстрая генерация без лишних операций"""
    try:
        model, tokenizer = load_model(mode)
        config = AGENT_CONFIG.get(mode, {"max_tokens": 128, "do_sample": False, "temperature": 1.0})
        
        # Системные промпты с явным указанием русского языка
        if mode == "orchestrator":
            system_msg = "Ты - помощник D&D мастера. ОТВЕТЬ НА РУССКОМ. Классифицируй действие на одну категорию: АТАКА, ДЕЙСТВИЕ, ДИАЛОГ или ИССЛЕДОВАНИЕ. Только категория."
        elif mode == "judge":
            system_msg = "Ты - Судья D&D 5e. ОТВЕТЬ НА РУССКОМ. Возможно ли действие? Ответь УСПЕХ или ПРОВАЛ. Если боевое - добавь урон (1d6+2)."
        elif mode == "keeper":
            system_msg = "Ты - хранитель памяти. ОТВЕТЬ НА РУССКОМ. Кратко (2-3 предложения): здоровье, локация, враги."
        else:  # storyteller
            system_msg = "Ты - мастер D&D. ОТВЕТЬ НА РУССКОМ от третьего лица (ВЫ, без 'я'). Опиши красиво (3-4 предложения). Атмосфера и действие."

        # Минимальный промпт
        if context:
            full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n[Контекст: {context[:150]}]\n{prompt[:150]}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt[:150]}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(TORCH_DEVICE)
        
        # Готовим параметры генерации - важно: top_k/top_p только если do_sample=True
        gen_kwargs = {
            "max_new_tokens": config["max_tokens"],
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": config["do_sample"],
            "temperature": config["temperature"],
            "repetition_penalty": config.get("repetition_penalty", 1.2)
        }
        
        # Добавляем top_k/top_p только если do_sample=True (иначе transformers выведет warning)
        if config["do_sample"]:
            gen_kwargs["top_k"] = config.get("top_k", 40)
            gen_kwargs["top_p"] = config.get("top_p", 0.9)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_text.split("assistant\n")[-1].strip()
        response_text = " ".join(response_text.splitlines()).strip()
        
        # Не обрезаем текст жестко - пусть модель определяет длину
        return response_text if response_text else "..."
    except Exception as e:
        print(f"ERROR in {mode}: {str(e)}")
        return "..."

@app.post("/generate")
async def generate_response(req: ChatRequest):
    """Быстрая синхронная генерация"""
    try:
        start = time.time()
        response = generate_fast(req.mode, req.prompt, req.context)
        elapsed = time.time() - start
        return {"response": response, "time": round(elapsed, 2)}
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/game/turn")
async def game_turn(req: ParallelRequest):
    """Синхронизированная обработка агентов с правильным потоком данных"""
    try:
        start_time = time.time()
        user_input = req.user_input[:200]
        world_state = req.world_state or {}
        context_str = str(world_state)[:300]
        
        loop = asyncio.get_event_loop()
        
        # === ШАГ 1: ОРКЕСТРАТОР - классифицирует намерение ===
        print(f"[1] Orchestrator: классифицирую '{user_input[:50]}...'")
        orch_result = await loop.run_in_executor(
            executor, generate_fast, "orchestrator", user_input, ""
        )
        print(f"    -> {orch_result}")
        
        # === ШАГ 2: СУДЬЯ - проверяет логичность (зависит от Оркестратора) ===
        judge_prompt = f"Действие игрока: '{user_input}'. Тип: {orch_result}. Текущее состояние: {context_str[:100]}"
        print(f"[2] Judge: проверяю логичность")
        judge_result = await loop.run_in_executor(
            executor, generate_fast, "judge", judge_prompt, context_str
        )
        print(f"    -> {judge_result}")
        
        # === ШАГ 3: KEEPER - обновляет память (параллельно) ===
        keeper_prompt = f"Действие: {user_input}. Исход: {judge_result}. Текущее состояние: {context_str}"
        print(f"[3] Keeper: обновляю память")
        keeper_result = await loop.run_in_executor(
            executor, generate_fast, "keeper", keeper_prompt, ""
        )
        print(f"    -> {keeper_result}")
        
        # === ШАГ 4: STORYTELLER - описывает сцену на основе Judge (зависит от Judge и Keeper) ===
        story_prompt = f"Действие: {user_input}. Исход: {judge_result}. Состояние: {keeper_result}. Опиши происходящее красиво, от третьего лица на ВЫ."
        print(f"[4] Storyteller: описываю сцену")
        storyteller_result = await loop.run_in_executor(
            executor, generate_fast, "storyteller", story_prompt, context_str
        )
        print(f"    -> {storyteller_result[:80]}...")
        
        elapsed = time.time() - start_time
        
        print(f"[✓] Turn completed in {elapsed:.2f}s")
        
        return {
            "status": "success",
            "time": round(elapsed, 2),
            "user_action": user_input,
            "classification": orch_result,
            "success": "УСПЕХ" in judge_result or "успех" in judge_result.lower(),
            "judge_decision": judge_result,
            "world_update": keeper_result,
            "narration": storyteller_result
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout - took over 12 seconds")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/logic")
async def process_logic(req: ChatRequest):
    # Внешний скрипт логики (без ИИ)
    result = DnDEngine.process_action(req.prompt)
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "cached": list(MODEL_CACHE.keys()), "device": DEVICE}

if __name__ == "__main__":
    import uvicorn

    try:
        port = PYTHON_SERVER_PORT
        if not is_port_free(PYTHON_SERVER_HOST, port):
            fallback_port = find_free_port(port + 1, port + 20)
            print(f"[!] Port {port} is busy. Using fallback port {fallback_port}.")
            print(f"    Set VITE_PYTHON_SERVER=http://localhost:{fallback_port} in .env if you need frontend connectivity.")
            port = fallback_port
        uvicorn.run(app, host=PYTHON_SERVER_HOST, port=port)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Unable to start server: {e}")
        sys.exit(1)
