"use client";

import { useState, useRef, useEffect } from "react";
import { askQuestion, getMessages, Citation, uploadDocuments } from "@/lib/api";
import { ArrowUp, Bot, Paperclip, AlertCircle, User, Info, Loader2 } from "lucide-react";

interface ChatAreaProps {
  convId: string | null;
}

export default function ChatArea({ convId }: ChatAreaProps) {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!convId) return;
    setMessages([]);
    getMessages(convId).then(setMessages).catch(console.error);
  }, [convId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || !convId || isLoading) return;

    const userMsg = { role: "user", content: input, id: Date.now().toString() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await askQuestion(convId, userMsg.content);
      const assistantMsg = {
        id: response.message_id,
        role: "assistant",
        content: response.answer_text,
        status: response.status,
        citations: response.citations,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setError(err.message || "خطا در دریافت پاسخ");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !convId) return;

    setIsUploading(true);
    setError(null);

    try {
      await uploadDocuments(convId, Array.from(files));
      // Tell the DocumentsPanel to refresh
      window.dispatchEvent(new Event("refresh_documents"));
    } catch (err: any) {
      setError(err.message || "خطا در آپلود فایل");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  if (!convId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-[#fafafa] text-gray-400">
        <Bot size={48} className="mb-4 text-gray-300" />
        <p>یک گفتگو را انتخاب کنید یا گفتگوی جدید بسازید.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-[#fafafa] relative">
      <div className="flex-1 overflow-y-auto p-4 md:px-8 pb-32 space-y-6">
        {messages.length === 0 && !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 mt-20">
            <div className="w-16 h-16 bg-teal-50 text-teal-600 rounded-2xl flex items-center justify-center mb-4 shadow-sm border border-teal-100">
              <Bot size={32} />
            </div>
            <h2 className="text-lg font-bold text-gray-700 mb-1">چطور می‌توانم کمک کنم؟</h2>
            <p className="text-sm">سوال خود را درباره اسناد بپرسید.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-4 max-w-3xl mx-auto ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 ${
              msg.role === "user" ? "bg-teal-600 text-white" : "bg-teal-50 text-teal-600 border border-teal-100"
            }`}>
              {msg.role === "user" ? <User size={16} /> : <Bot size={18} />}
            </div>
            
            <div className={`flex flex-col max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
              <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                msg.role === "user" 
                  ? "bg-teal-50 text-teal-900 border border-teal-100 rounded-tr-sm" 
                  : msg.status === "abstained"
                    ? "bg-gray-100 text-gray-600 border border-gray-200 rounded-tl-sm flex items-start gap-2"
                    : "bg-white border border-gray-100 shadow-sm rounded-tl-sm text-gray-800"
              }`}>
                {msg.status === "abstained" && <Info size={18} className="text-gray-400 shrink-0 mt-0.5" />}
                <div className="whitespace-pre-wrap">
                  {msg.role === "assistant" ? msg.content : msg.content}
                </div>
              </div>

              {msg.role === "assistant" && msg.status !== "abstained" && (
                <div className="mt-2 w-full space-y-2">
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-4 space-y-2">
                      <div className="text-xs font-medium text-gray-400 mb-2">منابع (برای مشاهده کلیک کنید):</div>
                      {msg.citations.map((c: Citation, i: number) => (
                        <CitationCard key={i} citation={c} />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 max-w-3xl mx-auto">
            <div className="w-8 h-8 rounded-full bg-teal-50 text-teal-600 border border-teal-100 flex items-center justify-center shrink-0">
              <Bot size={18} />
            </div>
            <div className="bg-white border border-gray-100 shadow-sm rounded-2xl rounded-tl-sm p-4 flex gap-1.5 items-center h-12">
              <div className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce"></div>
              <div className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: "0.15s" }}></div>
              <div className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: "0.3s" }}></div>
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-center max-w-3xl mx-auto">
            <div className="bg-red-50 border border-red-100 text-red-600 px-4 py-3 rounded-xl flex items-center gap-2 text-sm">
              <AlertCircle size={16} />
              {error}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#fafafa] via-[#fafafa] to-transparent">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white border border-gray-200 rounded-[24px] shadow-lg p-2 flex flex-col">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="پرسش‌تان را اینجا بنویسید..."
              className="w-full resize-none bg-transparent px-4 py-2 focus:outline-none text-sm text-gray-800 placeholder:text-gray-400 min-h-[44px] max-h-[200px]"
              rows={1}
              disabled={isLoading}
            />
            
            <div className="flex items-center justify-between px-2 pb-1 mt-1">
              <div className="flex items-center gap-1">
                <input
                  type="file"
                  accept=".pdf,.txt,.md"
                  multiple
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                />
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="text-gray-400 hover:text-teal-600 hover:bg-teal-50 p-2 rounded-xl transition-colors disabled:opacity-50"
                  title="پیوست سند جدید"
                >
                  {isUploading ? <Loader2 size={18} className="animate-spin text-teal-600" /> : <Paperclip size={18} />}
                </button>
              </div>
              
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="bg-teal-600 hover:bg-teal-700 text-white p-2 rounded-xl disabled:opacity-30 disabled:bg-gray-300 transition-all flex items-center justify-center shrink-0 w-10 h-10"
              >
                <ArrowUp size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2 text-gray-700">
          <div className="w-6 h-6 bg-teal-50 text-teal-600 rounded-md flex items-center justify-center">
            <Paperclip size={12} />
          </div>
          <span className="text-xs font-medium truncate max-w-[150px]">{citation.title}</span>
          <span className="text-[10px] bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full whitespace-nowrap">
            {citation.page_start ? `صفحه ${citation.page_start}` : "فایل متنی"}
          </span>
        </div>
        <span className="text-[10px] text-teal-600 font-medium">
          {isOpen ? "بستن" : "مشاهده متن"}
        </span>
      </button>
      
      {isOpen && (
        <div className="p-3 border-t border-gray-100 bg-[#fafafa]">
          {citation.breadcrumb && (
            <div className="text-xs text-gray-400 mb-2">{citation.breadcrumb}</div>
          )}
          <div className="text-gray-700 leading-relaxed text-xs border-r-2 border-teal-400 pr-3">
            {citation.quote}
            {citation.quote_truncated && <span className="text-gray-400">...</span>}
          </div>
        </div>
      )}
    </div>
  );
}