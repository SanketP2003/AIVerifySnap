import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/theme-provider";
import { ClerkProvider } from "@clerk/nextjs";
import { HeaderNav } from "@/components/shared/HeaderNav";

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
    <ClerkProvider>
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

            <HeaderNav />

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
    </ClerkProvider>
  );
}
