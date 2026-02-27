import { Badge } from "@/components/ui/badge";

interface ResultBadgeProps {
    status: "Real" | "AI Generated" | "Suspicious";
}

export function ResultBadge({ status }: ResultBadgeProps) {
    let badgeClass = "";
    if (status === "Real") badgeClass = "bg-green-500/10 text-green-600 hover:bg-green-500/20";
    else if (status === "AI Generated") badgeClass = "bg-destructive/10 text-destructive hover:bg-destructive/20";
    else badgeClass = "bg-yellow-500/10 text-yellow-600 hover:bg-yellow-500/20";

    return (
        <Badge variant="outline" className={`px-3 py-1 text-sm font-semibold rounded-full transition-colors border-0 ${badgeClass}`}>
            {status}
        </Badge>
    );
}
