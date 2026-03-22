"use client";

import React, { useState } from "react";
import { Send, Bot } from "lucide-react";

interface ChatAssistantProps {
  authHeaders: Record<string, string>;
  patientData: any;
}

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

export default function ChatAssistant({ authHeaders, patientData }: ChatAssistantProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm your HealthGuard AI assistant. Run a risk analysis first and then ask me anything about your results, lifestyle changes, or what your biomarkers mean.\n\n*Disclaimer: This assistant provides educational information only and does not replace professional medical advice.*",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({
          message: userMessage.content,
          patient_data: patientData || {
            predicted_risks: {},
            risk_level: "Unknown",
            top_risk_factors: [],
            protective_factors: [],
            patient_profile: {},
          },
          history: messages.slice(-6),
        }),
      });

      if (!res.ok) throw new Error("API Connection Error");
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.assistant_response || "Sorry, I couldn't process that." },
      ]);

      if (data.safety_flag) {
        setMessages((prev) => [
          ...prev,
          { role: "system", content: "⚠️ Safety notice: Please consult a healthcare professional for clinical decisions." },
        ]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Connection error: ${err.message}. Make sure the API server is running.` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-[#0f172a] border border-[#1e293b] rounded-3xl overflow-hidden shadow-2xl relative">
      <div className="flex items-center gap-4 bg-white/5 border-b border-white/5 p-4 sm:p-5 backdrop-blur-xl shrink-0">
        <div className="w-12 h-12 bg-gradient-to-br from-[#6c47ff] to-[#8b5cf6] rounded-2xl flex items-center justify-center text-white shadow-lg shadow-[#6c47ff]/20">
          <Bot size={24} />
        </div>
        <div>
          <h2 className="text-white font-bold text-base sm:text-lg">AI Health Assistant</h2>
          <div className="flex items-center gap-2 text-xs font-medium text-[#94a3b8]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.6)]"></span>
            Powered by Groq llama-3.1-8b
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 bg-gradient-to-b from-transparent to-black/20 scroll-smooth">
        {messages.map((m, i) => {
          if (m.role === "system") {
            return (
              <div key={i} className="text-center text-xs font-semibold text-amber-500/80 uppercase tracking-wide my-4 bg-amber-500/10 py-2 rounded-lg border border-amber-500/20">
                {m.content}
              </div>
            );
          }

          const isUser = m.role === "user";
          return (
            <div key={i} className={`flex ${isUser ? "justify-end" : "justify-start"} animate-in slide-in-from-bottom-2 duration-300 ease-out`}>
              <div
                className={`max-w-[85%] sm:max-w-[75%] rounded-2xl p-4 text-sm leading-relaxed shadow-md ${
                  isUser
                    ? "bg-[#6c47ff] text-white rounded-br-sm shadow-[#6c47ff]/20"
                    : "bg-[#1e293b] text-[#f1f5f9] rounded-bl-sm border border-[#334155]"
                }`}
                dangerouslySetInnerHTML={{ __html: m.content.replace(/\n/g, "<br/>") }}
              />
            </div>
          );
        })}
        {loading && (
          <div className="flex justify-start animate-in fade-in duration-300">
            <div className="bg-[#1e293b] text-[#94a3b8] rounded-2xl rounded-bl-sm p-4 w-16 h-12 flex justify-center items-center gap-1.5 shadow-md border border-[#334155]">
              <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:-0.3s]"></span>
              <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:-0.15s]"></span>
              <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce"></span>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 bg-white/5 border-t border-white/5 backdrop-blur-xl shrink-0">
        <form onSubmit={sendMessage} className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your health risk…"
            className="w-full bg-[#1e293b] border border-[#334155] rounded-full pl-5 pr-14 py-3.5 text-white text-sm placeholder:text-[#64748b] focus:outline-none focus:border-[#6c47ff] focus:ring-1 focus:ring-[#6c47ff] transition-all"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 bg-[#6c47ff] text-white rounded-full flex items-center justify-center hover:bg-[#5b3ae6] transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
          >
            <Send size={18} className="translate-x-[1px]" />
          </button>
        </form>
      </div>
    </div>
  );
}
