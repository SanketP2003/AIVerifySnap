"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

// Suppress the harmless "Encountered a script tag" warning from next-themes.
// next-themes injects a <script> to prevent theme flash (FOUC), which React 19+
// flags but is functionally correct. This only runs in development.
if (typeof window !== "undefined" && process.env.NODE_ENV === "development") {
    const origConsoleError = console.error;
    console.error = (...args: unknown[]) => {
        if (
            typeof args[0] === "string" &&
            args[0].includes("Encountered a script tag")
        ) {
            return;
        }
        origConsoleError.apply(console, args);
    };
}

export function ThemeProvider({
    children,
    ...props
}: React.ComponentProps<typeof NextThemesProvider>) {
    return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
