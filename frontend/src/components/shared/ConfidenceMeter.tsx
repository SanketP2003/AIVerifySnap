import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

interface ConfidenceMeterProps {
    score: number; // 0 to 100
    label?: string;
    result?: "Real" | "AI Generated" | "Suspicious";
}

export function ConfidenceMeter({ score, label = "Confidence", result }: ConfidenceMeterProps) {
    const pathColor = result === "Real"
        ? '#22c55e'
        : result === "Suspicious"
            ? '#eab308'
            : '#ef4444';

    const getStatusText = (val: number) => {
        if (val >= 90) return "Very High Certainty";
        if (val >= 70) return "High Certainty";
        if (val >= 50) return "Moderate Certainty";
        return "Low Certainty";
    };

    return (
        <div
            className="flex flex-col items-center justify-center p-4"
            role="progressbar"
            aria-valuenow={score}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`${label}: ${score}%`}
        >
            <div className="w-32 h-32 mb-3 relative group">
                <div
                    className="absolute inset-0 rounded-full blur-md opacity-20 transition-opacity group-hover:opacity-40 pointer-events-none"
                    style={{ backgroundColor: pathColor }}
                />
                <CircularProgressbar
                    value={score}
                    text={`${score}%`}
                    styles={buildStyles({
                        textSize: '18px',
                        pathColor: pathColor,
                        textColor: 'var(--foreground)',
                        trailColor: 'var(--border)',
                        pathTransitionDuration: 1.2,
                        strokeLinecap: 'round',
                    })}
                />
            </div>
            <p className="text-sm font-semibold tracking-tight text-foreground">{label}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{getStatusText(score)}</p>
        </div>
    );
}
