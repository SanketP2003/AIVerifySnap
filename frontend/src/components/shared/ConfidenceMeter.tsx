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

    return (
        <div className="flex flex-col items-center justify-center p-4">
            <div className="w-32 h-32 mb-4">
                <CircularProgressbar
                    value={score}
                    text={`${score}%`}
                    styles={buildStyles({
                        textSize: '18px',
                        pathColor: pathColor,
                        textColor: 'var(--foreground)',
                        trailColor: 'var(--border)',
                        pathTransitionDuration: 1.5,
                    })}
                />
            </div>
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
        </div>
    );
}
