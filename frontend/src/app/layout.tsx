import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/theme-provider";
import { ClerkProvider } from "@clerk/nextjs";
import { WorkspaceShell } from "@/components/shared/WorkspaceShell";

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
          className="antialiased min-h-screen bg-background text-foreground"
        >
          <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
            {/* Skip to content (a11y) */}
            <a href="#main-content" className="skip-to-content">
              Skip to main content
            </a>

            <WorkspaceShell>
              {children}
            </WorkspaceShell>

            <Toaster position="top-right" richColors />
          </ThemeProvider>
        </body>
      </html>
    </ClerkProvider>
  );
}
