import type { Metadata } from "next";
import { Vazirmatn } from "next/font/google";
import "./globals.css";

const vazirmatn = Vazirmatn({ 
  subsets: ["arabic"],
  weight: ["300", "400", "500", "700", "800", "900"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "RPKRAG | دستیار هوشمند اسناد",
  description: "پرسش و پاسخ از اسناد سازمانی با استناد دقیق",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fa" dir="rtl">
      <body className={`${vazirmatn.className} antialiased h-screen overflow-hidden bg-gray-100 text-gray-800`}>
        {children}
      </body>
    </html>
  );
}