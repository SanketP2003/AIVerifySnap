"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function DashboardSettingsRedirect() {
    const router = useRouter();

    useEffect(() => {
        router.replace("/settings");
    }, [router]);

    return (
        <div className="p-8 text-center text-xs text-muted-foreground font-mono">
            Redirecting to Workspace Settings...
        </div>
    );
}
