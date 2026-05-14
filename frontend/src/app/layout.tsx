import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/theme-provider";
import { ClerkProvider } from "@clerk/nextjs";
import { HeaderNav } from "@/components/shared/HeaderNav";
import Link from "next/link";

export const metadata: Metadata = {
  title: "AIVerifySnap | Deepfake Verification Platform",
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
        <head>
          <link rel="preconnect" href="https://fonts.googleapis.com" />
          <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
          <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
        </head>
        <body className="antialiased min-h-screen flex flex-col bg-background text-foreground" suppressHydrationWarning>
          <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
            {/* Announcement Bar */}
            <div className="w-full py-2.5 flex items-center justify-center text-[13px] font-medium gap-2.5 px-4 relative z-50"
              style={{ background: "linear-gradient(135deg, #e17b27ff 0%, #ecac2cff 50%, #fe7200ff 100%)", color: "white" }}>
              <span className="px-2 py-0.5 rounded text-[11px] font-bold border border-white/25 bg-white/15">BETA</span>
              <span>Forensic Engine is live. Try it now</span>
              <span className="text-base leading-none">&rarr;</span>
            </div>

            <HeaderNav />

            <main className="flex-1">{children}</main>

            {/* Footer */}
            <footer className="w-full mt-32 border-t" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
              <div className="mx-auto max-w-[1280px] px-6 py-16">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-10 md:gap-8">
                  <div className="col-span-2 md:col-span-2">
                    <Link href="/" className="inline-block mb-3">
                      <span className="text-lg font-bold tracking-tighter lowercase text-foreground">
                        ai<span className="font-light text-accent">verify</span>snap
                      </span>
                    </Link>
                    <p className="text-[14px] text-muted-foreground leading-relaxed max-w-[280px]">
                      India&apos;s deepfake detection and digital identity protection platform.
                    </p>
                  </div>

                  <div>
                    <h4 className="text-[11px] font-semibold tracking-[0.15em] uppercase text-muted-foreground mb-4">Platform</h4>
                    <ul className="space-y-2.5">
                      <li><Link href="/detect" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Image Detection</Link></li>
                      <li><Link href="/detect?mode=bulk" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Bulk Scan</Link></li>
                      <li><Link href="/protection" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Identity Protection</Link></li>
                      <li><Link href="/history" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Scan History</Link></li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-[11px] font-semibold tracking-[0.15em] uppercase text-muted-foreground mb-4">Developers</h4>
                    <ul className="space-y-2.5">
                      <li><Link href="/docs" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">API Reference</Link></li>
                      <li><Link href="/#how-it-works" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">How It Works</Link></li>
                      <li><Link href="/admin" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Dashboard</Link></li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-[11px] font-semibold tracking-[0.15em] uppercase text-muted-foreground mb-4">Company</h4>
                    <ul className="space-y-2.5">
                      <li><Link href="/pricing" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Pricing</Link></li>
                      <li><Link href="/#about" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">About</Link></li>
                      <li><Link href="/#contact" className="text-[14px] text-foreground/60 hover:text-foreground transition-colors">Contact</Link></li>
                    </ul>
                  </div>
                </div>

                <div className="flex flex-col md:flex-row items-center justify-between pt-10 mt-12 gap-3" style={{ borderTop: "1px solid hsl(var(--border) / 0.3)" }}>
                  <p className="text-[12px] text-muted-foreground">
                    &copy; {new Date().getFullYear()} AIVerifySnap. All rights reserved.
                  </p>
                  <p className="text-[11px] text-muted-foreground/60 font-medium tracking-[0.2em] uppercase">
                    Truth starts here
                  </p>
                </div>
              </div>
            </footer>
            <Toaster />
          </ThemeProvider>
        </body>
      </html>
    </ClerkProvider>
  );
}
