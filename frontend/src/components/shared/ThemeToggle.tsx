"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

export function ThemeToggle({ className = "" }: { className?: string }) {
    const { setTheme, resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) {
        return (
            <button
                type="button"
                className={`p-2 rounded-full border border-border bg-card/60 text-muted-foreground transition-colors hover:text-foreground hover:bg-muted ${className}`}
                aria-label="Toggle theme"
            >
                <span className="w-4 h-4 inline-block" />
            </button>
        );
    }

    const isDark = resolvedTheme === "dark";

    return (
        <button
            type="button"
            onClick={() => setTheme(isDark ? "light" : "dark")}
            className={`p-2 rounded-full border border-border bg-card/70 hover:bg-muted text-foreground transition-all duration-200 focus-ring shadow-sm ${className}`}
            aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
            {isDark ? (
                <Sun className="w-4 h-4 text-amber-400 transition-transform hover:rotate-45" />
            ) : (
                <Moon className="w-4 h-4 text-foreground/80 transition-transform hover:-rotate-12" />
            )}
        </button>
    );
}
