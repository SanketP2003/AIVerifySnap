"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function DashboardAnalyzeRedirect() {
    const router = useRouter();
    useEffect(() => {
        router.replace("/detect");
    }, [router]);

    return (
        <div className="p-8 text-center text-xs text-muted-foreground font-mono">
            Redirecting to Image Authenticity Workspace...
        </div>
    );
}
