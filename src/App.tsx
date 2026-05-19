import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Terminal,
  Sparkles,
  Shield,
  Brain,
  Ghost
} from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [gameStarted, setGameStarted] = useState(false);
  const [inputMode, setInputMode] = useState<'action' | 'dialogue'>('action');
  const [worldState, setWorldState] = useState('Вы находитесь в таверне. Рядом тавернщик. Вы ищете Барняра.');
  const chatEndRef = useRef<HTMLDivElement>(null);

  const API_URL = "http://localhost:8000";

  // Локальный кеш истории для ИИ
  const getContext = () => {
    return messages.slice(-4).map(m => `${m.role === 'user' ? 'Игрок' : 'Мастер'}: ${m.content}`).join('\n');
  };

  const checkServer = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) setServerStatus('online');
      else setServerStatus('offline');
    } catch {
      setServerStatus('offline');
    }
  };

  useEffect(() => {
    checkServer();
    const timer = setInterval(checkServer, 10000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const callAgent = async (mode: string, prompt: string) => {
    setCurrentStep(mode === 'storyteller' ? 'Описываю...' : mode === 'judge' ? 'Проверяю правила...' : mode === 'keeper' ? 'Запоминаю...' : 'Анализирую...');
    const response = await fetch(`${API_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        mode, 
        prompt, 
        context: mode === 'keeper' ? getContext() : `Состояние мира: ${worldState}\n${getContext()}`,
        max_tokens: mode === 'storyteller' ? 512 : 256 
      })
    });
    if (!response.ok) throw new Error(`Agent ${mode} failed`);
    const data = await response.json();
    return data.response;
  };

  const startGame = () => {
    const lore = "Вы стоите перед тяжелыми дубовыми дверями таверны 'Хромой Дракон'. Ветер свистит в узких улочках теневого квартала, донося запахи печеного мяса и дешевого эля. Ваша цель — найти информатора по кличке Барняр. Что вы будете делать?";
    setMessages([{ id: 'init', role: 'assistant', content: lore }]);
    setGameStarted(true);
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userContent = inputMode === 'dialogue' ? `Я говорю: "${input}"` : input;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: userContent
    };

    setMessages(prev => [...prev, userMsg]);
    const userInput = userContent;
    setInput('');
    setIsLoading(true);

    try {
      // 1. СУДЬЯ: Анализ возможности действия
      // Если это разговор, судья почти всегда пропускает
      let canProceed = true;
      let promptToDescribe = userInput;

      if (inputMode === 'action') {
        const judgeVerdict = await callAgent('judge', userInput);
        if (judgeVerdict.toUpperCase().includes('НЕВОЗМОЖНО')) {
          canProceed = false;
          setMessages(prev => [...prev, {
            id: Date.now().toString(),
            role: 'assistant',
            content: "Вы пытаетесь сделать нечто невозможное. Мастер качает головой: 'Это за пределами ваших сил или законов этого мира'."
          }]);
        }
      }
      
      if (canProceed) {
        // 2. ЛОГИКА: Броски кубиков (только для действий)
        let contextForStory = userInput;
        if (inputMode === 'action') {
          setCurrentStep('Бросаю кубики...');
          const logicRes = await fetch(`${API_URL}/logic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'logic', prompt: userInput })
          });
          const logicData = await logicRes.json();
          contextForStory = `Игрок: ${userInput}. Итог механики: ${logicData.text}.`;
        }
        
        // 3. РАССКАЗЧИК: Превращаем итог в художественный текст
        const storyPrompt = `Действие: ${userInput}. Механический итог: ${inputMode === 'action' ? contextForStory : 'Успешный разговор'}. Опиши результат КРАСИВО и БЕЗ ЦИФР.`;
        const narrative = await callAgent('storyteller', storyPrompt);

        const assistantMsg: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: narrative
        };
        setMessages(prev => [...prev, assistantMsg]);

        // 4. ХРАНИТЕЛЬ: Обновляем состояние мира (в фоне)
        callAgent('keeper', "Обнови описание ситуации").then(newState => {
          if (newState) setWorldState(newState);
        }).catch(e => console.error("Keeper failed", e));
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, {
        id: 'err',
        role: 'assistant',
        content: "Мастер на мгновение задумался... (Ошибка связи. Проверьте сервер)."
      }]);
    } finally {
      setIsLoading(false);
      setCurrentStep(null);
    }
  };

  if (!gameStarted) {
    return (
      <div className="min-h-screen bg-[#090a0c] text-slate-200 flex flex-col items-center justify-center p-6 text-center">
        <Sparkles className="w-16 h-16 text-red-600 mb-8 animate-pulse" />
        <h1 className="text-4xl font-bold tracking-[0.3em] mb-4 uppercase">D&D CORE</h1>
        <p className="text-slate-500 max-w-md mb-12 font-serif italic">
          "Погрузитесь в мир, где каждое ваше слово меняет реальность, а искусственный интеллект плетет нити вашей судьбы."
        </p>
        <button 
          onClick={startGame}
          disabled={serverStatus !== 'online'}
          className="px-12 py-4 bg-red-600 hover:bg-red-500 disabled:opacity-20 text-white rounded-full font-bold tracking-widest transition-all shadow-[0_0_30px_rgba(220,38,38,0.3)]"
        >
          {serverStatus === 'online' ? 'НАЧАТЬ ПУТЕШЕСТВИЕ' : 'ОЖИДАНИЕ СЕРВЕРА...'}
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#090a0c] text-slate-200 font-sans flex flex-col items-center">
      {/* Статус-бар */}
      <div className="w-full h-1 bg-gray-900 shrink-0">
        <div className={`h-full transition-all duration-1000 ${serverStatus === 'online' ? 'bg-red-500 w-full' : 'bg-gray-800 w-1/12'}`} />
      </div>

      <div className="w-full max-w-3xl flex-1 flex flex-col relative overflow-hidden px-4">
        
        {/* Инструкция и состояние */}
        <div className="mt-4 flex flex-col gap-4">
          <div className="p-4 bg-white/5 border border-white/10 rounded-2xl text-[10px] text-slate-400 font-mono leading-relaxed">
            <span className="text-red-500 font-bold">[ИНСТРУКЦИЯ]</span><br/>
            - <span className="text-white">ДЕЙСТВИЕ</span>: броски, бой, движение, мысли.<br/>
            - <span className="text-white">РАЗГОВОР</span>: прямая речь (префикс добавится сам).<br/>
            - Мастер понимает команды "Вспомнить", "Осмотреться", "Проверить сумку".
          </div>
          
          <div className="p-4 bg-red-900/10 border border-red-900/30 rounded-2xl">
            <div className="text-[9px] font-bold text-red-500 mb-2 uppercase tracking-widest flex items-center gap-2">
              <Brain className="w-3 h-3" /> Текущая обстановка (Keeper)
            </div>
            <div className="text-xs text-slate-400 font-serif italic text-balance">
              {worldState}
            </div>
          </div>
        </div>
        
        {/* Шапка */}
        <div className="flex items-center justify-between py-8 border-b border-white/5">
          <div className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-red-500" />
            <h1 className="text-xl font-bold tracking-[0.2em] uppercase">D&D CORE</h1>
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => { if(confirm('Начать новую игру?')) setGameStarted(false); }}
              className="text-[9px] font-mono text-slate-500 hover:text-white transition-colors"
            >
              [СБРОС]
            </button>
            <div className={`text-[10px] font-mono px-3 py-1 rounded-full border transition-colors ${serverStatus === 'online' ? 'border-emerald-500 text-emerald-500 bg-emerald-500/5' : 'border-red-500 text-red-500 bg-red-500/5'}`}>
              {serverStatus === 'online' ? 'READY' : 'OFFLINE'}
            </div>
          </div>
        </div>

        {/* Чат */}
        <div className="flex-1 overflow-y-auto space-y-10 py-10 scrollbar-hide">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center opacity-30 space-y-6">
              <div className="p-4 rounded-full border border-white/10">
                <Ghost className="w-12 h-12" />
              </div>
              <div className="space-y-2">
                <p className="font-serif italic text-xl">"Мир рождается из ваших слов..."</p>
                <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500">Ожидание инициализации мастера</p>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-6 duration-700`}>
              <div className={`relative max-w-[90%] ${
                msg.role === 'user' 
                  ? 'bg-red-600/5 border border-red-600/20 p-6 rounded-2xl rounded-tr-none' 
                  : 'bg-white/[0.03] border border-white/10 p-8 rounded-3xl rounded-tl-none shadow-2xl'
              }`}>
                <div className="absolute -top-3 left-4 px-2 py-0.5 bg-[#090a0c] text-[8px] font-mono text-gray-500 uppercase tracking-widest">
                  {msg.role === 'user' ? 'Игрок' : 'Мастер'}
                </div>
                <div className={`leading-relaxed ${msg.role === 'assistant' ? 'font-serif text-slate-100 text-xl' : 'font-sans text-slate-300 text-lg'}`}>
                  {msg.content}
                </div>
              </div>
            </div>
          ))}

          {/* Анимация ожидания */}
          {isLoading && (
            <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2">
              <div className="bg-white/[0.02] border border-white/10 p-6 rounded-2xl flex items-center gap-4">
                <div className="relative flex items-center justify-center">
                  <div className="absolute w-8 h-8 rounded-full border border-red-500/30 animate-ping" />
                  <div className="w-8 h-8 rounded-full border-2 border-t-red-600 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] font-mono text-red-500 uppercase tracking-[0.3em] font-bold">
                    Мастер размышляет
                  </span>
                  <span className="text-[8px] text-gray-600 uppercase tracking-[0.2em]">
                    {currentStep || "Плетение судьбы..."}
                  </span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Зона ввода */}
        <div className="pb-12 pt-6 shrink-0">
          <div className="max-w-2xl mx-auto relative group">
            
            {/* Переключатель режимов */}
            <div className="flex gap-2 mb-4 justify-center relative z-20">
              <button 
                type="button"
                onClick={(e) => { e.preventDefault(); setInputMode('action'); }}
                className={`flex items-center gap-2 px-6 py-2 rounded-full text-[10px] font-bold tracking-widest transition-all ${inputMode === 'action' ? 'bg-red-600 text-white shadow-[0_0_15px_rgba(220,38,38,0.3)]' : 'bg-white/5 text-slate-500 hover:text-slate-300'}`}
              >
                <Shield className="w-3 h-3" /> ДЕЙСТВИЕ
              </button>
              <button 
                type="button"
                onClick={(e) => { e.preventDefault(); setInputMode('dialogue'); }}
                className={`flex items-center gap-2 px-6 py-2 rounded-full text-[10px] font-bold tracking-widest transition-all ${inputMode === 'dialogue' ? 'bg-red-600 text-white shadow-[0_0_15px_rgba(220,38,38,0.3)]' : 'bg-white/5 text-slate-500 hover:text-slate-300'}`}
              >
                <Terminal className="w-3 h-3" /> РАЗГОВОР
              </button>
            </div>

            <div className="absolute -inset-2 bg-red-600/5 blur-2xl opacity-0 group-focus-within:opacity-100 transition-opacity" />
            <div className="relative flex items-center bg-[#111318] border border-white/5 rounded-3xl pl-8 pr-3 py-3 shadow-2xl backdrop-blur-md">
              <input 
                type="text" 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder={inputMode === 'action' ? "Опишите ваше действие..." : "Что вы скажете?"}
                className="flex-1 bg-transparent border-none focus:ring-0 py-3 text-lg placeholder-slate-700 text-slate-100"
              />
              <button 
                onClick={handleSend}
                disabled={isLoading}
                className="w-14 h-14 flex items-center justify-center bg-red-600 hover:bg-red-500 disabled:opacity-20 disabled:hover:bg-red-600 text-white rounded-2xl transition-all shadow-xl"
              >
                <Send className="w-6 h-6" />
              </button>
            </div>
            <p className="text-center text-[9px] text-slate-600 mt-4 uppercase tracking-[0.3em] font-bold">
              Autonomous Multimodal Story-telling Engine
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
