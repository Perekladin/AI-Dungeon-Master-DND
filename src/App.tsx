import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Terminal,
  Sparkles,
  Shield,
  Brain,
  Ghost
} from 'lucide-react';
import { playTurn, type TurnHistoryEntry } from './lib/agents';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

const INITIAL_WORLD = {
  location: 'Таверна "Хромой Дракон" в теневом квартале',
  summary: 'Игрок ищет информатора по кличке Барняр. В зале тавернщик, у стола двое наёмников.',
  player: { hp_current: 24, hp_max: 24, modifier: 3 },
  entities: {},
  flags: {},
};

interface CharacterSheet {
  name: string;
  race: string;
  class: string;
  level: number;
  hp_current: number;
  hp_max: number;
  ac: number;
  abilities: Record<string, number>;
  ability_modifiers: Record<string, number>;
  proficiency_bonus: number;
  skills: Record<string, { proficient: boolean; bonus: number }>;
  weapons: Array<{ name: string; attack_bonus: number; damage_dice: string }>;
  spells: Array<{ name: string; level: number; effect?: string }>;
  inventory: string[];
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [gameStarted, setGameStarted] = useState(false);
  const [inputMode, setInputMode] = useState<'action' | 'dialogue'>('action');
  const [worldState, setWorldState] = useState<Record<string, any>>(INITIAL_WORLD);
  const [character, setCharacter] = useState<CharacterSheet | null>(null);
  const [showSheet, setShowSheet] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Авто-определение API URL: берём хост браузера + порт 8000.
  // Это позволяет открывать сайт по LAN (192.168.x.x:5173) или через туннель —
  // фронт всегда стучит на ту же машину, где живёт Python-сервер.
  const API_URL = (() => {
    if (typeof window !== "undefined" && window.location?.hostname) {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
    return "http://localhost:8000";
  })();

  const checkServer = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      setServerStatus(res.ok ? 'online' : 'offline');
    } catch {
      setServerStatus('offline');
    }
  };

  const loadCharacter = async () => {
    try {
      const res = await fetch(`${API_URL}/character`);
      if (res.ok) setCharacter(await res.json());
    } catch (e) {
      console.warn('Не удалось загрузить чарник', e);
    }
  };

  useEffect(() => {
    checkServer();
    loadCharacter();
    const timer = setInterval(checkServer, 10000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const startGame = () => {
    const lore = "Вы стоите перед тяжелыми дубовыми дверями таверны 'Хромой Дракон'. Ветер свистит в узких улочках теневого квартала, донося запахи печеного мяса и дешевого эля. Ваша цель — найти информатора по кличке Барняр. Что вы будете делать?";
    setMessages([{ id: 'init', role: 'assistant', content: lore }]);
    setWorldState(INITIAL_WORLD);
    setGameStarted(true);
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userContent = inputMode === 'dialogue' ? `Я говорю: "${input}"` : input;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: userContent };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setCurrentStep('Плетение судьбы...');

    // Собираем историю последних 3 пар Игрок→Мастер для continuity Storyteller'а
    const history: TurnHistoryEntry[] = [];
    let pendingPlayer: string | null = null;
    for (const m of messages.slice(-8)) {
      if (m.role === 'user') {
        pendingPlayer = m.content;
      } else if (m.role === 'assistant' && pendingPlayer) {
        history.push({ player: pendingPlayer, master: m.content });
        pendingPlayer = null;
      }
    }

    try {
      const result = await playTurn(userContent, worldState, history);

      // Если Судья сказал impossible — кратко уведомляем игрока
      if (result.intent?.action === 'impossible') {
        setMessages(prev => [...prev, {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: result.narration || "Мастер качает головой: это за пределами законов этого мира.",
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: result.narration,
        }]);
      }

      // world_state — источник истины с сервера (HP, локация, флаги)
      setWorldState(result.world_state);

      // Перезагружаем чарник после хода — на случай если игрок получил урон от контратаки
      // (CHARACTER.hp_current обновляется на сервере в apply_state_delta).
      loadCharacter();

      console.log(`[turn] ${result.time}s category=${result.category} roll=${result.mechanics.roll} success=${result.mechanics.success}`);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, {
        id: 'err' + Date.now(),
        role: 'assistant',
        content: "Мастер на мгновение задумался... (Ошибка связи. Проверьте сервер)."
      }]);
    } finally {
      setIsLoading(false);
      setCurrentStep(null);
    }
  };

  const worldSummary = worldState?.summary || worldState?.location || 'Мир ждёт ваших действий.';

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
              {worldSummary}
            </div>
          </div>

          {/* Панель противников: live HP с цветовым индикатором.
              Без этой панели игрок не понимает что один наёмник умер и спавнился новый — у него ощущение «дерусь долго с одним». */}
          {(() => {
            const entities = worldState?.entities || {};
            const alive = Object.entries(entities).filter(
              ([, e]: any) => e && e.hp_current > 0
            );
            const dead = Object.entries(entities).filter(
              ([, e]: any) => e && e.hp_current <= 0
            );
            if (alive.length === 0 && dead.length === 0) return null;
            return (
              <div className="p-4 bg-slate-900/40 border border-slate-700/40 rounded-2xl">
                <div className="text-[9px] font-bold text-amber-500 mb-3 uppercase tracking-widest flex items-center gap-2">
                  ⚔️ Противники в сцене
                </div>
                <div className="space-y-2">
                  {alive.map(([eid, e]: any) => {
                    const hpRatio = e.hp_current / (e.hp_max || 1);
                    const barColor = hpRatio > 0.6 ? 'bg-emerald-500' : hpRatio > 0.3 ? 'bg-amber-500' : 'bg-red-500';
                    return (
                      <div key={eid} className="text-xs">
                        <div className="flex justify-between font-mono text-slate-300 mb-0.5">
                          <span>{e.name || eid} <span className="text-slate-500">[{eid}]</span></span>
                          <span>HP {e.hp_current}/{e.hp_max}</span>
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
                          <div
                            className={`h-full ${barColor} transition-all duration-500`}
                            style={{ width: `${hpRatio * 100}%` }}
                          />
                        </div>
                        <div className="text-[10px] text-slate-600 mt-0.5 font-mono">
                          AC {e.ac}, поведение: {e.behavior_tag}
                        </div>
                      </div>
                    );
                  })}
                  {dead.length > 0 && (
                    <div className="text-[10px] text-slate-500 font-mono pt-2 border-t border-slate-700/30">
                      💀 Поверженных: {dead.map(([eid, e]: any) => e.name || eid).join(', ')}
                    </div>
                  )}
                </div>
              </div>
            );
          })()}
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
              {character && (
                <button
                  type="button"
                  onClick={(e) => { e.preventDefault(); setShowSheet(s => !s); }}
                  className={`flex items-center gap-2 px-6 py-2 rounded-full text-[10px] font-bold tracking-widest transition-all border ${showSheet ? 'bg-amber-600/20 text-amber-300 border-amber-500/50' : 'bg-white/5 text-amber-500/70 border-amber-500/20 hover:text-amber-400'}`}
                >
                  📜 {character.name} · HP {character.hp_current}/{character.hp_max}
                </button>
              )}
            </div>

            {/* Панель чарника — раскрывается над полем ввода */}
            {showSheet && character && (
              <div className="mb-4 p-5 bg-amber-950/10 border border-amber-900/30 rounded-2xl font-mono text-xs text-amber-100 space-y-3 animate-in fade-in slide-in-from-bottom-2">
                <div className="flex justify-between border-b border-amber-900/30 pb-2">
                  <span className="font-bold tracking-widest">{character.name.toUpperCase()}</span>
                  <span className="text-amber-400">{character.race} {character.class} — ур. {character.level}</span>
                </div>
                <div className="grid grid-cols-3 gap-3 text-[11px]">
                  <div><span className="text-amber-500">HP</span> {character.hp_current}/{character.hp_max}</div>
                  <div><span className="text-amber-500">AC</span> {character.ac}</div>
                  <div><span className="text-amber-500">Маст.</span> +{character.proficiency_bonus}</div>
                </div>
                <div className="grid grid-cols-6 gap-2 text-[10px]">
                  {Object.entries(character.ability_modifiers).map(([k, v]) => (
                    <div key={k} className="text-center border border-amber-900/30 rounded p-1">
                      <div className="text-amber-500 uppercase">{k}</div>
                      <div className="text-amber-100">{v >= 0 ? `+${v}` : v}</div>
                    </div>
                  ))}
                </div>
                <div className="text-[11px]">
                  <span className="text-amber-500">Оружие:</span>{' '}
                  {character.weapons.map(w => `${w.name} (атака +${w.attack_bonus}, урон ${w.damage_dice})`).join('; ')}
                </div>
                {character.spells.length > 0 && (
                  <div className="text-[11px]">
                    <span className="text-amber-500">Заклинания:</span>{' '}
                    {character.spells.map(s => `${s.name} (ур. ${s.level})`).join('; ')}
                  </div>
                )}
                <div className="text-[11px]">
                  <span className="text-amber-500">Тренир. навыки:</span>{' '}
                  {Object.entries(character.skills).filter(([_, s]) => s.proficient).map(([n, s]) => `${n} +${s.bonus}`).join(', ')}
                </div>
              </div>
            )}

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
