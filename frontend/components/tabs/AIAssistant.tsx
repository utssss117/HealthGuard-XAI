'use client';

import { useState, useRef, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { sendChatMessage, type Biomarkers, type PredictResponse, type ChatMessage } from '@/lib/api';

type Props = {
  biomarkers: Biomarkers | null;
  prediction: PredictResponse | null;
};

const SUGGESTIONS = [
  'What does my glucose level mean?',
  'How can I lower my diabetes risk?',
  'What dietary changes do you recommend?',
  'Explain my top risk factors.',
];

export default function AIAssistantTab({ biomarkers, prediction }: Props) {
  const { getToken } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const send = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: ChatMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError(null);

    if (!prediction) {
      setTimeout(() => {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Please run a Risk Assessment first so I can give you context-aware health guidance based on your specific biomarkers.',
        }]);
        setLoading(false);
      }, 300);
      return;
    }

    try {
      const token = await getToken();
      const res = await sendChatMessage({
        message: text,
        patient_data: {
          patient_profile: biomarkers,
          predicted_risks: { diabetes: prediction.risk_probability },
          risk_level: prediction.risk_level,
          top_risk_factors: [],
          protective_factors: [],
        },
        history: messages.slice(-10),
      }, token);

      let reply = res.assistant_response;
      if (res.safety_flag) reply = '🛡️ **Safety Notice**\n\n' + reply;
      if (res.escalation_required) reply += '\n\n---\n🚨 Please contact emergency services or a physician immediately.';

      setMessages(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Could not reach the AI assistant. Please check the API connection.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="glass-panel p-4 flex items-center justify-between">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-xl filter drop-shadow-[0_0_8px_rgba(59,130,246,0.2)]">🤖</div>
          <div>
            <h2 className="font-bold text-white text-sm">HealthGuard AI Assistant</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Groq LLaMA-3.1 · {prediction ? `${prediction.risk_level} risk context loaded` : 'No context loaded — run assessment first'}
            </p>
          </div>
        </div>
        {messages.length > 0 && (
          <button
            id="btn-clear-chat"
            onClick={() => setMessages([])}
            className="text-xs text-slate-400 hover:text-white border border-slate-800 hover:bg-white/[0.03] px-3.5 py-1.5 rounded-lg transition-colors cursor-pointer"
          >
            Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="glass-panel min-h-[350px] max-h-[480px] overflow-y-auto p-5 flex flex-col gap-4">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center py-12 text-slate-500">
            <span className="text-5xl mb-4 opacity-75 filter drop-shadow-[0_0_15px_rgba(59,130,246,0.15)]">💬</span>
            <p className="text-sm text-center max-w-xs text-slate-400">
              Ask any health question regarding your biomarkers, risks, or daily routine.
            </p>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[75%] px-4.5 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-br-none shadow-[0_4px_12px_rgba(59,130,246,0.15)]'
                    : 'bg-slate-900 border border-slate-800/80 text-slate-200 rounded-bl-none'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-900 border border-slate-800/80 text-slate-400 px-4.5 py-3 rounded-2xl text-sm rounded-bl-none flex items-center gap-2">
              <span className="w-3.5 h-3.5 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin" />
              Thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestion chips */}
      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2.5">
          {SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              id={`suggestion-${i}`}
              onClick={() => send(s)}
              className="border border-blue-500/10 text-blue-400 hover:text-blue-300 hover:bg-blue-500/10 bg-blue-500/5 text-xs px-4 py-2 rounded-full transition-all duration-200 cursor-pointer font-medium"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div className="text-xs text-rose-400 bg-rose-500/5 border border-rose-500/10 rounded-xl px-4 py-3">{error}</div>
      )}

      {/* Input */}
      <div className="flex gap-3">
        <textarea
          id="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input); } }}
          placeholder="Ask a health question… (Enter to send, Shift+Enter for new line)"
          rows={2}
          className="input-premium flex-1 px-4 py-3 text-sm resize-none focus:outline-none"
        />
        <button
          id="btn-send"
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
          className="btn-premium px-6 flex items-center justify-center transition-all duration-200 cursor-pointer disabled:opacity-50"
        >
          ➤
        </button>
      </div>
    </div>
  );
}
