"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function DashboardHistoryRedirect() {
    const router = useRouter();
    useEffect(() => {
        router.replace("/history");
    }, [router]);

    return (
        <div className="p-8 text-center text-xs text-muted-foreground font-mono">
            Redirecting to Forensic Scan History...
        </div>
    );
}
