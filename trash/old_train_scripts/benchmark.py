#!/usr/bin/env python3
"""
Скрипт для бенчмаркинга DND AI - проверка скорости обработки
"""

import asyncio
import time
import json
import httpx
from typing import Dict, List

PYTHON_SERVER = "http://localhost:8000"
EXPRESS_SERVER = "http://localhost:3000"
TIMEOUT_SECONDS = 12

test_inputs = [
    "Я атакую орка мечом! Выпало 18!",
    "Я хочу понять, что это за руны на стене",
    "Я говорю с торговцем о цене зелий",
    "Я исследую комнату, ищу сокровища",
    "Я использую магию огня на драконе! Выпало 16!",
]

async def test_python_server():
    """Тест прямого вызова Python сервера"""
    print("\n" + "="*60)
    print("🔧 Тест Python сервера (FastAPI)")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        for input_text in test_inputs:
            print(f"\n📝 Input: {input_text}")
            start = time.time()
            
            try:
                response = await client.post(
                    f"{PYTHON_SERVER}/game/turn",
                    json={"user_input": input_text, "world_state": {}}
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Success ({data['time']:.2f}s)")
                    print(f"  Orchestrator: {data['orchestrator'][:50]}")
                    print(f"  Judge: {data['judge'][:50]}")
                else:
                    print(f"✗ Error {response.status_code}: {response.text[:100]}")
            except Exception as e:
                elapsed = time.time() - start
                print(f"✗ Failed after {elapsed:.2f}s: {str(e)[:100]}")

async def test_express_server():
    """Тест Express сервера (полный цикл)"""
    print("\n" + "="*60)
    print("🚀 Тест Express сервера (полный цикл)")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        for input_text in test_inputs:
            print(f"\n📝 Input: {input_text}")
            start = time.time()
            
            try:
                response = await client.post(
                    f"{EXPRESS_SERVER}/api/game/turn",
                    json={"userInput": input_text}
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    total_time = data.get('timing', {}).get('total_api', elapsed)
                    print(f"✓ Success ({total_time:.2f}s)")
                    print(f"  Category: {data.get('category')}")
                    print(f"  Damage: {data.get('damage', 0)}")
                else:
                    print(f"✗ Error {response.status_code}: {response.text[:100]}")
            except Exception as e:
                elapsed = time.time() - start
                print(f"✗ Failed after {elapsed:.2f}s: {str(e)[:100]}")

async def test_individual_agents():
    """Тест каждого агента отдельно"""
    print("\n" + "="*60)
    print("🎭 Тест отдельных агентов")
    print("="*60)
    
    agents = ["orchestrator", "judge", "keeper", "storyteller"]
    test_prompt = "Я атакую орка мечом!"
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        for agent in agents:
            print(f"\n{agent.upper()}")
            print("-" * 40)
            
            timings = []
            for i in range(3):
                start = time.time()
                try:
                    response = await client.post(
                        f"{PYTHON_SERVER}/generate",
                        json={"mode": agent, "prompt": test_prompt, "context": ""}
                    )
                    elapsed = time.time() - start
                    
                    if response.status_code == 200:
                        data = response.json()
                        timings.append(data['time'])
                        if i == 0:  # Показываем только первый результат
                            print(f"Response: {data['response'][:60]}")
                    else:
                        print(f"Error: {response.status_code}")
                except Exception as e:
                    print(f"Error: {str(e)[:60]}")
            
            if timings:
                avg_time = sum(timings) / len(timings)
                min_time = min(timings)
                max_time = max(timings)
                print(f"Timings: min={min_time:.2f}s, avg={avg_time:.2f}s, max={max_time:.2f}s")

async def test_parallel_inference():
    """Тест параллельной обработки"""
    print("\n" + "="*60)
    print("⚡ Тест параллельной обработки (4 агента одновременно)")
    print("="*60)
    
    test_prompt = "Я атакую орка мечом! Выпало 18!"
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        start = time.time()
        
        tasks = [
            client.post(f"{PYTHON_SERVER}/generate", json={"mode": "orchestrator", "prompt": test_prompt, "context": ""}),
            client.post(f"{PYTHON_SERVER}/generate", json={"mode": "judge", "prompt": test_prompt, "context": ""}),
            client.post(f"{PYTHON_SERVER}/generate", json={"mode": "keeper", "prompt": test_prompt, "context": ""}),
            client.post(f"{PYTHON_SERVER}/generate", json={"mode": "storyteller", "prompt": test_prompt, "context": ""}),
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start
        
        print(f"\n✓ All 4 agents processed in {total_time:.2f}s")
        
        for i, response in enumerate(responses):
            if not isinstance(response, Exception) and response.status_code == 200:
                data = response.json()
                print(f"  Agent {i+1}: {data['time']:.2f}s")

async def main():
    print("\n" + "="*60)
    print("🧪 DND AI Performance Benchmark")
    print("="*60)
    print(f"Timeout: {TIMEOUT_SECONDS}s")
    print(f"Python Server: {PYTHON_SERVER}")
    print(f"Express Server: {EXPRESS_SERVER}")
    
    # Проверяем доступность серверов
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{PYTHON_SERVER}/health")
            print(f"✓ Python server: OK ({r.status_code})")
        except:
            print(f"✗ Python server: NOT RESPONDING")
            return
        
        try:
            r = await client.get(f"{EXPRESS_SERVER}/api/health")
            print(f"✓ Express server: OK ({r.status_code})")
        except:
            print(f"⚠ Express server: NOT RESPONDING (optional)")
    
    # Запускаем тесты
    await test_individual_agents()
    await test_parallel_inference()
    await test_python_server()
    
    if True:  # Если доступен Express
        await test_express_server()
    
    print("\n" + "="*60)
    print("✅ Benchmark complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
