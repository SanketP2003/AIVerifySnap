"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function DashboardReportsRedirect() {
    const router = useRouter();

    useEffect(() => {
        router.replace("/reports");
    }, [router]);

    return (
        <div className="p-8 text-center text-xs text-muted-foreground font-mono">
            Redirecting to Forensic Reports Dossiers...
        </div>
    );
}
