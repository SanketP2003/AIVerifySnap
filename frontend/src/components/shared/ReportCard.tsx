import { ResultBadge } from "./ResultBadge";
import { ConfidenceMeter } from "./ConfidenceMeter";

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
    return (
        <div className="flex flex-col rounded-3xl border border-border bg-card p-6 shadow-sm shadow-black/5 hover:border-primary/20 transition-all duration-300">
            <h2 className="text-xl font-bold tracking-tight mb-6">Forensic Analysis</h2>

            <div className="flex flex-col sm:flex-row items-center justify-between space-y-4 sm:space-y-0 pb-6 mb-6 border-b border-border">
                <div className="flex flex-col">
                    <p className="text-sm text-balance text-muted-foreground mb-1">Detection Result</p>
                    <div className="flex items-center space-x-3">
                        <h3 className={`text-2xl font-bold ${result === "Real" ? "text-green-600" : result === "Suspicious" ? "text-yellow-600" : "text-destructive"}`}>
                            {result}
                        </h3>
                        <ResultBadge status={result} />
                    </div>
                </div>
                <ConfidenceMeter score={confidence} result={result} />
            </div>

            <div className="flex flex-col space-y-4 text-sm font-medium">
                <div className="flex justify-between items-center bg-muted/30 p-3 rounded-xl border border-border/50">
                    <span className="text-muted-foreground tracking-tight">ELA Score Match</span>
                    <span className="font-bold text-foreground">{elaScore}</span>
                </div>
                <div className="flex justify-between items-center bg-muted/30 p-3 rounded-xl border border-border/50">
                    <span className="text-muted-foreground tracking-tight">Forensic Model Used</span>
                    <span className="font-bold text-foreground">{modelName}</span>
                </div>
                <div className="flex justify-between items-center bg-muted/30 p-3 rounded-xl border border-border/50">
                    <span className="text-muted-foreground tracking-tight">Time Generated</span>
                    <span className="text-foreground">{timestamp}</span>
                </div>
            </div>
        </div>
    );
}
