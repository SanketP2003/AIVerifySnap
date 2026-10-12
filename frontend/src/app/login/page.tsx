"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
    const router = useRouter();

    useEffect(() => {
        router.replace("/sign-in");
    }, [router]);

    return (
        <div className="min-h-[60vh] flex flex-col items-center justify-center p-6 text-center space-y-3">
            <div className="w-8 h-8 border-2 border-foreground border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                Connecting to Secure Authentication Gateway...
            </p>
        </div>
    );
}
