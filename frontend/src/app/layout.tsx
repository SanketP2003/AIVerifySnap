import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import Link from "next/link";
import { ChevronDown, Menu } from "lucide-react";
import { ThemeProvider } from "@/components/theme-provider";

// Replicating Sarvam Header
export const metadata: Metadata = {
  title: "AIVerifySnap | Sovereign Deepfake Verification Platform",
  description: "AI-Powered Deepfake Image Verification & Victim Protection System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`antialiased min-h-screen flex flex-col bg-background text-foreground`}
      >
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
          {/* Top Announcement Bar */}
          <div className="w-full bg-gradient-to-r from-[#e7af55] via-[#e57a40] to-[#e7af55] py-2 flex items-center justify-center text-white text-sm font-medium gap-2 px-4 shadow-sm z-50 relative">
            <span className="bg-white/20 px-2 py-0.5 rounded-md text-xs font-bold border border-white/30 backdrop-blur-sm">NEW</span>
            <span>Forensic Engine is live in beta. Try Now</span>
            <span className="text-lg leading-none">&rarr;</span>
          </div>

          {/* Main Navigation Header */}
          <header className="sticky top-0 z-40 w-full backdrop-blur-3xl bg-white/70 border-b border-black/5 support-[backdrop-filter]:bg-white/60">
            <div className="mx-auto max-w-[1400px] px-6">
              <div className="flex h-[72px] items-center justify-between">

                {/* Logo */}
                <Link href="/" className="flex items-center space-x-2 transition-opacity hover:opacity-70">
                  <span className="font-bold tracking-tighter text-[26px] lowercase text-[#131313]">ai<span className="font-light">verify</span>snap</span>
                </Link>

                {/* Desktop Nav Links */}
                <nav className="hidden lg:flex items-center gap-10">
                  <Link href="/detect" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                    Platform <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                  </Link>
                  <Link href="/history" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                    History <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                  </Link>
                  <Link href="/protection" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                    Protection <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                  </Link>
                  <Link href="/admin" className="flex items-center gap-1.5 text-[13px] font-semibold tracking-widest uppercase text-[#131313] hover:opacity-70 transition-opacity">
                    Admin <ChevronDown className="w-4 h-4 text-black/40" strokeWidth={3} />
                  </Link>
                </nav>

                {/* CTA Buttons */}
                <div className="hidden lg:flex items-center gap-4">
                  <Link href="/detect" className="flex items-center justify-center px-6 py-2.5 text-[15px] font-medium rounded-full bg-[#1e1e1e] text-white hover:bg-[#2b2b2b] shadow-[inset_0_1px_4px_rgba(255,255,255,0.4)] transition-all duration-300">
                    Experience Platform
                  </Link>
                  <button className="flex items-center justify-center px-6 py-2.5 text-[15px] font-medium rounded-full bg-black/5 text-black hover:bg-black/10 transition-colors border border-black/5">
                    Talk to Sales
                  </button>
                </div>

                {/* Mobile Menu Icon */}
                <button className="lg:hidden p-2 text-black">
                  <Menu className="w-6 h-6" />
                </button>
              </div>
            </div>
          </header>

          <main className="flex-1">
            {children}
          </main>

          <footer className="w-full bg-[#f4f4f4] mt-24">
            <div className="mx-auto max-w-[1400px] px-6 py-24 flex flex-col items-center text-center">
              <h2 className="text-3xl md:text-5xl font-serif text-[#131313] mb-8">Build the future of digital trust<br />with AIVerifySnap.</h2>
              <p className="text-sm font-semibold tracking-widest uppercase text-black/40">Truth starts here</p>
            </div>
          </footer>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
