"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";
import DocumentsPanel from "@/components/DocumentsPanel";
import PrivacyModal from "@/components/PrivacyModal";
import { BookOpen, User, PanelRightClose, PanelRightOpen } from "lucide-react";

export default function Home() {
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [showDocs, setShowDocs] = useState(false); // Start closed for better animation

  useEffect(() => {
    const ack = localStorage.getItem("privacy_ack");
    if (!ack) setShowPrivacy(true);
    setIsReady(true);
  }, []);

  const handleAcceptPrivacy = () => {
    localStorage.setItem("privacy_ack", "true");
    setShowPrivacy(false);
  };

  if (!isReady) return null;

  const isDocsOpen = showDocs && activeConvId !== null;

  return (
    <>
      {showPrivacy && <PrivacyModal onAccept={handleAcceptPrivacy} />}
      
      <div className="flex h-screen w-full bg-gray-100 overflow-hidden relative">
        
        {/* Main Content (Sidebar + Chat) */}
        {/* Added blur, scale, and opacity transitions when docs panel is open */}
        <div className={`flex flex-1 min-w-0 transition-all duration-500 ease-in-out ${isDocsOpen ? 'blur-[2px] scale-[0.99] opacity-70 pointer-events-none' : ''}`}>
          
          <Sidebar 
            activeConvId={activeConvId} 
            onSelectConv={(id) => setActiveConvId(id)} 
          />
          
          <main className="flex-1 flex flex-col min-h-0 bg-gray-50">
            <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 px-6 py-3 flex items-center justify-between z-10 shrink-0">
              
              {/* Right side of header */}
              <div className="flex items-center gap-2">
                <BookOpen className="text-teal-600" size={22} />
                <h1 className="text-base font-bold text-gray-800 tracking-tight">RPKRAG</h1>
              </div>
              
              {/* Left side of header */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium text-gray-700">کاربر</div>
                  <div className="w-8 h-8 bg-gray-200 text-gray-600 rounded-full flex items-center justify-center border border-gray-300">
                    <User size={16} />
                  </div>
                </div>

                {/* Documents Toggle Button moved here */}
                <div className="w-px h-6 bg-gray-300 mx-1"></div>
                <button 
                  onClick={() => setShowDocs(!showDocs)}
                  disabled={!activeConvId}
                  className={`p-2 rounded-lg transition-all flex items-center gap-2 text-sm font-medium border ${
                    isDocsOpen 
                      ? "bg-teal-50 border-teal-200 text-teal-700 shadow-sm" 
                      : "bg-white border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-teal-600"
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                  title="نمایش/مخفی کردن اسناد"
                >
                  {isDocsOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
                  <span className="hidden sm:inline">اسناد</span>
                </button>
              </div>

            </header>

            <ChatArea convId={activeConvId} />
          </main>
        </div>

        {/* Documents Panel (Animates Width) */}
        <div className={`h-full shrink-0 transition-all duration-300 ease-in-out overflow-hidden ${isDocsOpen ? 'w-[340px] opacity-100' : 'w-0 opacity-0'}`}>
          <div className="w-[340px] h-full">
            {isDocsOpen && (
              <DocumentsPanel convId={activeConvId} onClose={() => setShowDocs(false)} />
            )}
          </div>
        </div>

      </div>
    </>
  );
}