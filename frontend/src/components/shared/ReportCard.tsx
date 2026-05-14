import { ResultBadge } from "./ResultBadge";
import { ConfidenceMeter } from "./ConfidenceMeter";

interface ReportCardProps {
    result: "Real" | "AI Generated" | "Suspicious";
    confidence: number;
    elaScore: string;
    modelName: string;
    timestamp: string;
}

export function ReportCard({ result, confidence, elaScore, modelName, timestamp }: ReportCardProps) {
    const rows = [
        { label: "ELA Score Match", value: elaScore },
        { label: "Forensic Model", value: modelName },
        { label: "Timestamp", value: timestamp },
    ];

    return (
        <div className="card-elevated rounded-2xl p-6">
            <h2 className="text-heading font-semibold text-foreground mb-5">Forensic Analysis</h2>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pb-5 mb-5" style={{ borderBottom: "1px solid hsl(var(--border) / 0.4)" }}>
                <div>
                    <p className="text-caption text-muted-foreground mb-1">Detection Result</p>
                    <div className="flex items-center gap-2.5">
                        <span className={`text-xl font-bold ${result === "Real" ? "text-green-600" : result === "Suspicious" ? "text-orange-600" : "text-destructive"}`}>
                            {result}
                        </span>
                        <ResultBadge status={result} />
                    </div>
                </div>
                <ConfidenceMeter score={confidence} result={result} />
            </div>

            <div className="space-y-2.5">
                {rows.map((row) => (
                    <div key={row.label} className="flex justify-between items-center py-2.5 px-3.5 rounded-lg" style={{ background: "hsl(var(--muted) / 0.3)" }}>
                        <span className="text-caption text-muted-foreground">{row.label}</span>
                        <span className="text-[13px] font-semibold text-foreground">{row.value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
