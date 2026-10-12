import { ResultBadge } from "./ResultBadge";
import { ConfidenceMeter } from "./ConfidenceMeter";
import { Cpu, Clock, FileCheck2, Fingerprint } from "lucide-react";

interface ReportCardProps {
    result: "Real" | "AI Generated" | "Suspicious";
    confidence: number;
    elaScore: string;
    modelName: string;
    timestamp: string;
}

export function ReportCard({
    result,
    confidence,
    elaScore,
    modelName,
    timestamp,
}: ReportCardProps) {
    const resultColor =
        result === "Real"
            ? "text-green-600"
            : result === "Suspicious"
                ? "text-yellow-600"
                : "text-destructive";

    return (
        <div className="flex flex-col rounded-3xl border border-border bg-card p-6 sm:p-7 shadow-sm hover:border-primary/20 transition-all duration-300">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-border">
                <div className="flex items-center gap-2">
                    <Fingerprint className="w-5 h-5 text-primary" />
                    <h2 className="text-xl font-bold tracking-tight">Forensic Analysis</h2>
                </div>
                <ResultBadge status={result} size="md" />
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-6 pb-6 mb-6 border-b border-border">
                <div className="flex flex-col text-center sm:text-left">
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                        Detection Verdict
                    </p>
                    <h3 className={`text-3xl font-black tracking-tight ${resultColor}`}>
                        {result}
                    </h3>
                    <p className="text-xs text-muted-foreground mt-2 max-w-xs">
                        {result === "Real"
                            ? "Natural compression signatures and authentic camera artifacts detected."
                            : result === "AI Generated"
                                ? "Synthetic frequency anomalies and neural generation traces identified."
                                : "Inconclusive artifacts present. Manual forensic verification recommended."}
                    </p>
                </div>
                <div className="shrink-0">
                    <ConfidenceMeter score={confidence} result={result} label="Analysis Score" />
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
                <div className="flex flex-col justify-between bg-muted/40 p-3.5 rounded-2xl border border-border/60">
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-medium mb-1.5">
                        <FileCheck2 className="w-3.5 h-3.5 text-primary" />
                        <span>ELA Match Score</span>
                    </div>
                    <span className="font-semibold text-foreground text-xs sm:text-sm font-mono truncate" title={elaScore}>
                        {elaScore}
                    </span>
                </div>

                <div className="flex flex-col justify-between bg-muted/40 p-3.5 rounded-2xl border border-border/60">
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-medium mb-1.5">
                        <Cpu className="w-3.5 h-3.5 text-primary" />
                        <span>Inference Engine</span>
                    </div>
                    <span className="font-semibold text-foreground text-xs sm:text-sm truncate" title={modelName}>
                        {modelName}
                    </span>
                </div>

                <div className="flex flex-col justify-between bg-muted/40 p-3.5 rounded-2xl border border-border/60">
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-medium mb-1.5">
                        <Clock className="w-3.5 h-3.5 text-primary" />
                        <span>Timestamp</span>
                    </div>
                    <span className="font-semibold text-foreground text-xs sm:text-sm truncate" title={timestamp}>
                        {timestamp}
                    </span>
                </div>
            </div>
        </div>
    );
}
