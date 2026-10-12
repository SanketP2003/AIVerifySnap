import { Badge } from "@/components/ui/badge";
import { CheckCircle2, AlertTriangle, ShieldAlert } from "lucide-react";

interface ResultBadgeProps {
    status: "Real" | "AI Generated" | "Suspicious";
    size?: "sm" | "md" | "lg";
    showIcon?: boolean;
}

export function ResultBadge({ status, size = "md", showIcon = true }: ResultBadgeProps) {
    let badgeClass = "";
    let Icon = CheckCircle2;
    let dotClass = "";

    if (status === "Real") {
        badgeClass = "bg-green-500/10 text-green-600 hover:bg-green-500/20 border-green-500/20";
        Icon = CheckCircle2;
        dotClass = "bg-green-500";
    } else if (status === "AI Generated") {
        badgeClass = "bg-destructive/10 text-destructive hover:bg-destructive/20 border-destructive/20";
        Icon = ShieldAlert;
        dotClass = "bg-destructive";
    } else {
        badgeClass = "bg-yellow-500/10 text-yellow-600 hover:bg-yellow-500/20 border-yellow-500/20";
        Icon = AlertTriangle;
        dotClass = "bg-yellow-500";
    }

    const sizeClasses = {
        sm: "px-2.5 py-0.5 text-xs gap-1",
        md: "px-3 py-1 text-sm gap-1.5",
        lg: "px-4 py-1.5 text-base gap-2 font-bold",
    };

    const iconSizes = {
        sm: "w-3 h-3",
        md: "w-3.5 h-3.5",
        lg: "w-4 h-4",
    };

    return (
        <Badge
            variant="outline"
            className={`inline-flex items-center font-semibold rounded-full transition-all border ${badgeClass} ${sizeClasses[size]}`}
        >
            <span className={`w-1.5 h-1.5 rounded-full ${dotClass} animate-pulse`} aria-hidden="true" />
            {showIcon && <Icon className={`${iconSizes[size]} shrink-0`} aria-hidden="true" />}
            <span>{status}</span>
        </Badge>
    );
}
