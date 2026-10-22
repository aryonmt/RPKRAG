"use client";

import { useState, useEffect, useRef } from "react";
import { Document, getDocuments, uploadDocuments, deleteDocument } from "@/lib/api";
import { UploadCloud, FileText, Loader2, CheckCircle2, AlertCircle, Trash2, X, Plus } from "lucide-react";

interface DocumentsPanelProps {
  convId: string | null;
  onClose: () => void;
}

export default function DocumentsPanel({ convId, onClose }: DocumentsPanelProps) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDocs = async () => {
    if (!convId) return;
    try {
      const data = await getDocuments(convId);
      setDocs(data);
    } catch (e) {
      console.error("Failed to load docs", e);
    }
  };

  useEffect(() => {
    if (!convId) {
      setDocs([]);
      return;
    }
    
    setDocs([]);
    setIsUploading(false);
    setUploadError(null);
    setIsDragOver(false);

    loadDocs();

    // Listen to custom event from ChatArea
    const handleRefresh = () => loadDocs();
    window.addEventListener("refresh_documents", handleRefresh);

    const interval = setInterval(() => {
      setDocs((currentDocs) => {
        const hasProcessing = currentDocs.some((d) => d.status === "processing");
        if (hasProcessing) loadDocs();
        return currentDocs;
      });
    }, 2000);

    return () => {
      clearInterval(interval);
      window.removeEventListener("refresh_documents", handleRefresh);
    };
  }, [convId]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !convId) return;
    await upload(Array.from(files));
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (!files || files.length === 0 || !convId) return;
    await upload(Array.from(files));
  };

  const upload = async (files: File[]) => {
    setIsUploading(true);
    setUploadError(null);

    try {
      await uploadDocuments(convId!, files);
      await loadDocs(); 
    } catch (err: any) {
      setUploadError(err.message || "خطا در آپلود فایل‌ها");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = ""; 
    }
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!convId) return;
    if (!window.confirm(`آیا از حذف سند "${filename}" مطمئن هستید؟`)) return;

    try {
      await deleteDocument(convId, docId);
      await loadDocs();
    } catch (err: any) {
      alert(err.message || "خطا در حذف سند");
    }
  };

  if (!convId) return null;

  return (
    <div className="w-full bg-white border-l border-gray-200 flex flex-col h-full shadow-[-4px_0_15px_-3px_rgba(0,0,0,0.05)] z-20">
      <div className="p-5 flex items-center justify-between border-b border-gray-100">
        <div className="flex items-center gap-2">
          <FileText size={18} className="text-teal-600" />
          <h2 className="text-sm font-bold text-gray-800">اسناد این گفتگو</h2>
        </div>
        <button 
          onClick={onClose} 
          className="text-gray-400 hover:text-red-500 hover:bg-red-50 p-1.5 rounded-lg transition-colors"
        >
          <X size={20} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5">
        <div className="flex flex-col items-center justify-center mb-6 mt-2">
          <div className="w-16 h-16 bg-teal-50 rounded-2xl flex items-center justify-center text-teal-600 relative mb-3 border border-teal-100 shadow-sm">
            <FileText size={28} />
            <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-teal-600 text-white rounded-full flex items-center justify-center border-2 border-white">
              <Plus size={14} />
            </div>
          </div>
          <h3 className="text-sm font-bold text-gray-800">دانش شما، نقطه شروع</h3>
          <p className="text-xs text-gray-500 text-center mt-1 leading-relaxed">
            مقاله، گزارش یا یادداشت خود را اضافه کنید و درباره‌اش بپرسید.
          </p>
        </div>

        <input
          type="file"
          accept=".pdf,.txt,.md"
          multiple
          className="hidden"
          ref={fileInputRef}
          onChange={handleFileChange}
        />
        
        <div 
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          className={`w-full border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all mb-4 ${
            isDragOver ? "border-teal-500 bg-teal-50" : "border-gray-300 hover:border-teal-400 hover:bg-gray-50"
          }`}
        >
          {isUploading ? (
            <Loader2 size={28} className="text-teal-600 animate-spin mb-2" />
          ) : (
            <UploadCloud size={28} className="text-gray-400 mb-2" />
          )}
          <span className="text-sm font-medium text-gray-700">
            {isUploading ? "در حال آپلود..." : "فایل‌ها را اینجا رها کنید"}
          </span>
          <span className="text-xs text-teal-600 mt-1 font-medium underline decoration-teal-300 underline-offset-4">
            یا یک یا چند سند انتخاب کنید
          </span>
          <span className="text-[10px] text-gray-400 mt-3 text-center">
            سند، متن، داده و تصویر • هر فایل تا ۸۰ MB
          </span>
        </div>
        
        {uploadError && (
          <div className="mb-4 text-xs text-red-500 text-center flex items-center justify-center gap-1 bg-red-50 p-2 rounded-lg border border-red-100">
            <AlertCircle size={14} />
            {uploadError}
          </div>
        )}

        <div className="space-y-2 mt-6">
          {docs.map((doc) => (
            <div key={doc.id} className="p-3 bg-white rounded-xl border border-gray-100 group relative hover:border-gray-200 transition-colors">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-gray-50 flex items-center justify-center shrink-0 border border-gray-100">
                  <FileText size={14} className="text-gray-500" />
                </div>
                <div className="flex-1 min-w-0 pr-6">
                  <div className="text-xs text-gray-800 font-medium truncate" title={doc.filename}>
                    {doc.filename}
                  </div>
                  <div className="flex items-center gap-2 mt-1 text-[10px]">
                    <StatusBadge status={doc.status} />
                    {doc.chunk_count > 0 && (
                      <span className="text-gray-400">• {doc.chunk_count} تکه</span>
                    )}
                  </div>
                </div>
              </div>
              
              <button
                onClick={() => handleDelete(doc.id, doc.filename)}
                className="absolute top-1/2 -translate-y-1/2 left-2 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                title="حذف سند"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "processing") {
    return <span className="flex items-center gap-1 text-teal-600"><Loader2 size={10} className="animate-spin" /> در حال پردازش</span>;
  }
  if (status === "indexed" || status === "partial_warning") {
    return <span className="flex items-center gap-1 text-emerald-600"><CheckCircle2 size={10} /> آماده</span>;
  }
  if (status === "failed") {
    return <span className="flex items-center gap-1 text-red-500"><AlertCircle size={10} /> خطا</span>;
  }
  return <span className="text-gray-500">{status}</span>;
}