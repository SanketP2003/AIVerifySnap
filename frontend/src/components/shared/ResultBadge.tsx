interface ResultBadgeProps {
    status: "Real" | "AI Generated" | "Suspicious";
}

export function ResultBadge({ status }: ResultBadgeProps) {
    const styles: Record<string, string> = {
        "Real": "bg-green-500/8 text-green-600 dark:text-green-400",
        "AI Generated": "bg-destructive/8 text-destructive",
        "Suspicious": "bg-orange-500/8 text-orange-600 dark:text-orange-400",
    };

    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-[12px] font-semibold rounded-md transition-colors ${styles[status] || ""}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${
                status === "Real" ? "bg-green-500" : status === "AI Generated" ? "bg-destructive" : "bg-orange-500"
            }`} />
            {status}
        </span>
    );
}
