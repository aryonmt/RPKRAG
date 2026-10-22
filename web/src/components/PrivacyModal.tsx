"use client";

import { ShieldAlert } from "lucide-react";

interface PrivacyModalProps {
  onAccept: () => void;
}

export default function PrivacyModal({ onAccept }: PrivacyModalProps) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full overflow-hidden animate-slide-in">
        <div className="bg-amber-50 p-6 flex flex-col items-center text-center border-b border-amber-100">
          <div className="w-16 h-16 bg-amber-100 text-amber-600 rounded-full flex items-center justify-center mb-4">
            <ShieldAlert size={32} />
          </div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">هشدار حفظ حریم خصوصی</h2>
          <p className="text-gray-600 leading-relaxed text-sm">
            متن بخش‌های مرتبط از اسناد شما برای تولید پاسخ به سرویس مدل زبانی ارسال می‌شود. اسناد محرمانه را بدون بررسی سیاست داده‌ی سرویس ارسال نکنید.
          </p>
        </div>
        <div className="p-4 bg-gray-50 flex justify-end">
          <button
            onClick={onAccept}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm"
          >
            متوجه شدم و موافقم
          </button>
        </div>
      </div>
    </div>
  );
}