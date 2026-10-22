"use client";

import { useEffect, useState, useRef } from "react";
import { Conversation, getConversations, createConversation, deleteConversation, renameConversation } from "@/lib/api";
import { Plus, Search, MessageSquare, Trash2, Pencil } from "lucide-react";

interface SidebarProps {
  activeConvId: string | null;
  onSelectConv: (id: string | null) => void;
}

export default function Sidebar({ activeConvId, onSelectConv }: SidebarProps) {
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  
  // States for editing
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const data = await getConversations();
        setConvs(data);
        if (data.length > 0 && !activeConvId) {
          onSelectConv(data[0].id);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };

    fetchInitial();

    const interval = setInterval(async () => {
      try {
        const data = await getConversations();
        setConvs(data);
      } catch (e) {
        console.error(e);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Focus input when editing starts
  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
    }
  }, [editingId]);

  const handleCreate = async () => {
    try {
      const newConv = await createConversation("گفتگوی جدید");
      setConvs([newConv, ...convs]);
      onSelectConv(newConv.id);
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async (e: React.MouseEvent, convId: string, title: string) => {
    e.stopPropagation();
    if (!window.confirm(`آیا از حذف گفتگوی "${title}" و تمام اسناد آن مطمئن هستید؟`)) return;

    try {
      await deleteConversation(convId);
      if (activeConvId === convId) onSelectConv(null);
      
      const data = await getConversations();
      setConvs(data);
    } catch (e: any) {
      alert(e.message || "خطا در حذف گفتگو");
    }
  };

  const startEditing = (e: React.MouseEvent, conv: Conversation) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const saveRename = async (convId: string) => {
    if (!editTitle.trim()) {
      setEditingId(null); // Cancel if empty
      return;
    }
    
    try {
      await renameConversation(convId, editTitle.trim());
      // Update local state immediately
      setConvs(convs.map(c => c.id === convId ? { ...c, title: editTitle.trim() } : c));
    } catch (e: any) {
      alert(e.message || "خطا در تغییر نام");
    } finally {
      setEditingId(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent, convId: string) => {
    if (e.key === "Enter") {
      saveRename(convId);
    } else if (e.key === "Escape") {
      setEditingId(null);
    }
  };

  const filteredConvs = convs.filter(c => c.title.includes(searchQuery));

  return (
    <div className="w-[280px] bg-white border-l border-gray-100 flex flex-col h-full shrink-0">
      <div className="p-5 pb-2">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-teal-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
              د
            </div>
            <div>
              <h1 className="font-bold text-gray-800 text-sm">دستیار هوشمند</h1>
              <p className="text-[10px] text-gray-400">فضای شخصی شما</p>
            </div>
          </div>
        </div>

        <button
          onClick={handleCreate}
          className="w-full flex items-center justify-between bg-teal-50 hover:bg-teal-100 text-teal-700 py-2.5 px-3 rounded-xl transition-colors font-medium border border-teal-100"
        >
          <div className="flex items-center gap-2">
            <Plus size={18} />
            <span className="text-sm">گفتگوی تازه</span>
          </div>
          <span className="text-[10px] text-teal-600/60 font-mono dir-ltr">Ctrl+Shift+O</span>
        </button>

        <div className="mt-4 relative">
          <Search size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="جستجو در گفتگوها"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-gray-50 border border-gray-200 rounded-xl py-2 pr-9 pl-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition-all"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        <div className="text-xs text-gray-400 font-medium mb-2 px-2 mt-2">امروز</div>
        
        {loading ? (
          <div className="text-center text-gray-400 mt-4 text-sm">در حال بارگذاری...</div>
        ) : filteredConvs.length === 0 ? (
          <div className="text-center text-gray-400 mt-4 text-sm">گفتگویی یافت نشد.</div>
        ) : (
          filteredConvs.map((c) => (
            <div
              key={c.id}
              onClick={() => {
                if (editingId !== c.id) onSelectConv(c.id);
              }}
              className={`w-full text-right flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all cursor-pointer group relative ${
                activeConvId === c.id
                  ? "bg-teal-50/80 text-teal-800 font-medium"
                  : "hover:bg-gray-100 text-gray-700"
              }`}
            >
              <MessageSquare size={16} className={activeConvId === c.id ? "text-teal-600" : "text-gray-400"} />
              
              <div className="flex-1 min-w-0 pr-12">
                {editingId === c.id ? (
                  <input
                    ref={inputRef}
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onBlur={() => saveRename(c.id)}
                    onKeyDown={(e) => handleKeyDown(e, c.id)}
                    className="w-full bg-white border border-teal-300 rounded px-1.5 py-0.5 text-sm focus:outline-none focus:ring-1 focus:ring-teal-500"
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <div className="truncate text-sm">{c.title}</div>
                )}
              </div>
              
              {/* Action Buttons */}
              <div className="absolute top-1/2 -translate-y-1/2 left-2 flex items-center opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-all bg-white/90 rounded-lg shadow-sm border border-gray-100">
                <button
                  onClick={(e) => startEditing(e, c)}
                  className="p-1.5 text-gray-400 hover:text-teal-600 hover:bg-teal-50 rounded-r-lg transition-colors"
                  title="تغییر نام"
                >
                  <Pencil size={14} />
                </button>
                <div className="w-px h-3 bg-gray-200"></div>
                <button
                  onClick={(e) => handleDelete(e, c.id, c.title)}
                  className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-l-lg transition-colors"
                  title="حذف گفتگو"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

    </div>
  );
}