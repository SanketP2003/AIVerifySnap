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
            ? "text-green-600 dark:text-green-400"
            : result === "Suspicious"
                ? "text-yellow-600 dark:text-yellow-400"
                : "text-destructive";

    return (
        <div className="flex flex-col rounded-2xl border border-border bg-card p-5 sm:p-6 shadow-sm hover:border-foreground/20 transition-all duration-300">
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-border/80">
                <div className="flex items-center gap-2">
                    <Fingerprint className="w-4 h-4 text-primary" />
                    <h2 className="text-base sm:text-lg font-bold tracking-tight font-sans">Forensic Analysis Summary</h2>
                </div>
                <ResultBadge status={result} size="md" />
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-6 pb-6 mb-5 border-b border-border/80">
                <div className="flex flex-col text-center sm:text-left">
                    <p className="text-[11px] font-mono uppercase tracking-wider text-muted-foreground mb-1 font-semibold">
                        Detection Verdict
                    </p>
                    <h3 className={`text-2xl sm:text-3xl font-black tracking-tight ${resultColor}`}>
                        {result}
                    </h3>
                    <p className="text-xs text-muted-foreground mt-2 max-w-sm leading-relaxed">
                        {result === "Real"
                            ? "Natural compression signatures and authentic camera optical characteristics detected."
                            : result === "AI Generated"
                                ? "Synthetic frequency anomalies and neural generation traces identified."
                                : "Inconclusive artifacts present. Manual forensic verification recommended."}
                    </p>
                </div>
                <div className="shrink-0">
                    <ConfidenceMeter score={confidence} result={result} label="Analysis Score" />
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                <div className="flex flex-col justify-between bg-muted/40 p-3 rounded-xl border border-border/60">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium mb-1">
                        <FileCheck2 className="w-3.5 h-3.5 text-primary" />
                        <span>ELA Match Score</span>
                    </div>
                    <span className="font-semibold text-foreground font-mono truncate" title={elaScore}>
                        {elaScore}
                    </span>
                </div>

                <div className="flex flex-col justify-between bg-muted/40 p-3 rounded-xl border border-border/60">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium mb-1">
                        <Cpu className="w-3.5 h-3.5 text-primary" />
                        <span>Inference Engine</span>
                    </div>
                    <span className="font-semibold text-foreground truncate" title={modelName}>
                        {modelName}
                    </span>
                </div>

                <div className="flex flex-col justify-between bg-muted/40 p-3 rounded-xl border border-border/60">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium mb-1">
                        <Clock className="w-3.5 h-3.5 text-primary" />
                        <span>Timestamp</span>
                    </div>
                    <span className="font-semibold text-foreground font-mono text-[11px] truncate" title={timestamp}>
                        {timestamp}
                    </span>
                </div>
            </div>
        </div>
    );
}
